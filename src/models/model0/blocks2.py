import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import sqlite3
import pandas as pd
from itertools import islice
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.optimizer import Optimizer
import joblib
from torch.utils.data import Dataset, DataLoader, IterableDataset
import json
from torch.distributed import init_process_group, destroy_process_group
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CausalSelfAttention(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        assert model_config.n_embd % model_config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.Q = nn.Linear(model_config.n_embd, model_config.n_embd)
        self.K = nn.Linear(model_config.n_embd, model_config.n_embd)
        self.V = nn.Linear(model_config.n_embd, model_config.n_embd)

        self.token_to_a2 = nn.Linear(model_config.n_embd, model_config.squares_size * model_config.total_tokens)
        # output projection
        self.c_proj = nn.Linear(model_config.n_embd, model_config.n_embd)
        # regularization
        self.n_head = model_config.n_head
        self.n_embd = model_config.n_embd
        self.n_square_tokens = model_config.squares_size
        self.rpe_q = torch.tensor(model_config.squares_size, model_config.n_embd)
        self.rpe_k = torch.tensor(model_config.squares_size, model_config.n_embd)
        self.rpe_v = torch.tensor(model_config.squares_size, model_config.n_embd)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        q, k, v = self.Q(x) + self.rpe_q, self.K(x) + self.rpe_k, self.V(x) + self.rpe_v
        q = q.view(B, self.n_square_tokens, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        a1 = q @ k.transpose(-2,-1) # (B, nh, T2, T)
        board_overview_token = x[:, 0:1, :].squeeze(1) #(B, n_embd)
        a2 = self.token_to_a2(board_overview_token).view(B, self.n_head, self.n_square_tokens, T) # (B, nh, T2, T)
        att = a1 + a2 # (B, nh, T2, T)
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T2, hs)
        #y = F.scaled_dot_product_attention(q, k, v) # flash attention, is_causal=False cos no ,masking
        y = y.transpose(1, 2).contiguous().view(B, self.n_square_tokens, C) # re-assemble all head outputs side by side, (B, T2, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.c_fc    = nn.Linear(model_config.n_embd, 4 * model_config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * model_config.n_embd, model_config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_config.n_embd)
        self.attn = CausalSelfAttention(model_config)
        self.dropout_1 = nn.Dropout(model_config.dropout)
        self.ln_2 = nn.LayerNorm(model_config.n_embd)
        self.mlp = MLP(model_config)
        self.dropout_2 = nn.Dropout(model_config.dropout)

    def forward(self, x):
        x = x + self.dropout_1(self.attn(self.ln_1(x)))
        x = x + self.dropout_2(self.mlp(self.ln_2(x)))
        return x






class PolicyHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.fc = nn.Linear(model_config.n_embd, model_config.n_possible_moves) # Output full policy
        self.fc2 = nn.Linear(model_config.n_embd, model_config.n_moves_reevaluate) # Output reevaluation policy


    def forward(self, x, forward_pass="first", masked_indices=None):
        #B, T, C = x.shape
        if forward_pass == "first":
            x = self.fc(x)
        else:
            x = self.fc2(x)
        if masked_indices is not None:
            x = self.masked_softmax(x, masked_indices)
        return x

    def masked_softmax(self, x, masked_indices):  # masked_indices are the legal move indices
        # Create a mask with -inf for all positions that should be masked
        mask = torch.full_like(x, float("-inf"))
        
        # Filter out invalid indices (-1) from masked_indices in one go using vectorized operations
        valid_mask = masked_indices != -1  # Create a boolean mask for valid indices
        
        # Use `scatter_` to fill in 0 where the indices are valid
        mask.scatter_(1, masked_indices.clamp(0) * valid_mask, 0)
        
        # Add mask to input x
        masked_input = x + mask
        return masked_input

    def masked_softmax2(self, x, masked_indices): #actually masked_indices are the legal move indices that are not masked
        mask = torch.full_like(x, float("-inf"))
        for b in range(x.shape[0]): #): TODO check if change from x.shape -> x.shape[0] is correct
            batch_tensor = masked_indices[b]
            for i, value in enumerate(batch_tensor):
                if value == -1:
                    batch_tensor = batch_tensor[:i]
                    break

            #if masked_indices[b].item() != -1:
            mask[b, batch_tensor] = 0
        masked_input = x + mask
        output = masked_input
        return output

class Transformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.transformer = nn.ModuleDict(dict(
            te = nn.Embedding(model_config.vocab_size, model_config.n_embd),
            pe = nn.Embedding(model_config.squares_size, model_config.n_embd),
            me = nn.Embedding(model_config.n_possible_moves + 1, model_config.n_embd),
            mre = nn.Embedding(model_config.n_moves_reevaluate, model_config.n_embd), # move rank embeddings
            #se = nn.Embedding(1, model_config.n_embd), #scaling embedding
            h = nn.ModuleList([Block(model_config) for _ in range(model_config.n_layer)]),
            ln_f = nn.LayerNorm(model_config.n_embd),
        ))
        self.se = nn.Parameter(torch.randn(model_config.n_embd)).to(device)

    def forward(self, board_state_tensor, special_tokens_tensor, reevaluation_moves_tensor=None):
        # idx is of shape (B, T)
        B, T = board_state_tensor.size()
        assert T <= self.model_config.squares_size, f"Cannot forward sequence of length {T}, block size is only {self.model_config.squares_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=board_state_tensor.device)  # shape (T)
        pos_emb = self.transformer.pe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.te(board_state_tensor)  # token embeddings of shape (B, T, n_embd)
        special_emb = special_tokens_tensor.unsqueeze(-1).expand(B, self.model_config.special_size, self.model_config.n_embd)
        moves_tensor = torch.full((B, self.model_config.n_moves_reevaluate), self.model_config.n_possible_moves).to(device)
        move_emb = self.transformer.me(moves_tensor)


        x = torch.cat((tok_emb + pos_emb, special_emb, move_emb), dim=1)

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        self.out = x
        return self.out