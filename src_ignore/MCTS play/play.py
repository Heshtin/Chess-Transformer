import os
import math
import time
import chess
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
from torch.utils.data import Dataset, DataLoader

existing_model_path = "../runs/full_TCEC_run_1/iters/state_dict_v175.pth"



@dataclass
class HyperParamConfig:
    total_batch_size: int = 4096
    adamw_weight_decay: float = 1e-3
    gradient_clipping: float = 1.0
    max_lr: float = 1e-2
    max_steps: float = 0.80
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 128
    n_blocks_policyhead: int = 3
    n_blocks_valuehead: int = 4
    dropout: float = 0.1
class CausalSelfAttention(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        assert model_config.n_embd % model_config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(model_config.n_embd, 3 * model_config.n_embd)
        # output projection
        self.c_proj = nn.Linear(model_config.n_embd, model_config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = model_config.n_head
        self.n_embd = model_config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v) # flash attention, is_causal=False cos no ,masking
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.c_fc    = nn.Linear(model_config.n_embd, 4 * model_config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * model_config.n_embd, model_config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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


class LinBlock(nn.Module):
    def __init__(self, model_config, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=model_config.dropout)

    def forward(self, x):
        x = self.lin(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        #no residual connection bcos output size is different from input size
        return x


class ValueHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.blocks = nn.ModuleList()
        # self.reduce_linear = LinBlock(model_config, in_channels=model_config.n_embd*64, out_channels=model_config.n_embd)
        # self.blocks.append(self.reduce_linear)
        inchannels = model_config.n_embd
        log_reduction = int(math.log2(model_config.n_embd))
        common_reduction = log_reduction // model_config.n_blocks_valuehead
        n_additional_reductions = log_reduction % model_config.n_blocks_valuehead
        
        for i in range(model_config.n_blocks_valuehead):
            reduction = common_reduction if n_additional_reductions+i < model_config.n_blocks_valuehead else common_reduction + 1
            outchannels = inchannels // 2**(reduction)
            self.blocks.append(LinBlock(model_config, in_channels=inchannels*64, out_channels=outchannels*64))
            inchannels = outchannels
        self.fc = nn.Linear(64, 3)
        self.output_tanh = nn.Tanh()

    def forward(self, x):
        B, T, C = x.shape
        
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        #print(f"x:{x.shape=}")
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)  # Fully connected layer to output a scalar
        #x = self.output_tanh(x*0.1)
        return x





class PolicyHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.blocks = nn.ModuleList()
        # self.reduce_linear = LinBlock(model_config, in_channels=model_config.n_embd*64, out_channels=model_config.n_embd)
        # self.blocks.append(self.reduce_linear)

        inchannels = model_config.n_embd
        log_reduction = int(math.log2(model_config.n_embd)) - 5 # 2**11=2048, closest power of 2 to 1968
        common_reduction = log_reduction // model_config.n_blocks_policyhead
        n_additional_reductions = log_reduction % model_config.n_blocks_policyhead
        for i in range(int(model_config.n_blocks_policyhead)):
            reduction = common_reduction if n_additional_reductions+i < model_config.n_blocks_policyhead else common_reduction + 1
            outchannels = inchannels // 2**(reduction)
            self.blocks.append(LinBlock(model_config, in_channels=inchannels*64, out_channels=outchannels*64))
            inchannels = outchannels

        self.fc = nn.Linear(2048, 1968)  # Fully connected layer to output a scalar


    def forward(self, x, masked_indices):
        B, T, C = x.shape
        
        x = x.reshape(B, -1)
        for block in self.blocks:
            x = block(x)
        #print(f"{x.shape=}")
        x = self.fc(x)
        #x = self.masked_softmax(x, masked_indices)
        return x

    def masked_softmax(self, x, masked_indices):
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
            wte = nn.Embedding(model_config.vocab_size, model_config.n_embd),
            wpe = nn.Embedding(model_config.block_size, model_config.n_embd),
            h = nn.ModuleList([Block(model_config) for _ in range(model_config.n_layer)]),
            ln_f = nn.LayerNorm(model_config.n_embd),
        ))

    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.model_config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.model_config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        self.out = x
        return self.out


class Chess(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.transformer = Transformer(self.model_config)
        self.value_head = ValueHead(self.model_config)
        self.policy_head = PolicyHead(self.model_config)
        if existing_model_path is not None:
            print("loading existing state_dict")
            print(f"{existing_model_path=}")
            state_dict = torch.load(existing_model_path)
            adjusted_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()} #torch.compile is adding the prefix "_orig_mod." to all keys of the state_dict for some reason, need to remove it
            self.load_state_dict(adjusted_state_dict)
        else:
            print("initialising new model")
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            in_features = module.weight.size(1)  # Getting the size of input features
            a = 1 / math.sqrt(in_features)  # Xavier initialization scale factor
            if module.out_features == 1968 or module.out_features == 1:
                nn.init.uniform_(module.weight, -a, a)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            else:
                nn.init.uniform_(module.weight, -2*a, 2*a)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -math.sqrt(1/module.embedding_dim), math.sqrt(1/module.embedding_dim))
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def forward(self, data, legal_indices, p=None, v=None):
        # idx is of shape (B, T)
        B, T = data.size()
        x = self.transformer(data)

        x_policy = self.policy_head(x, legal_indices)
        x_policy = F.softmax(x_policy, dim=1)
        x_value = self.value_head(x)
        x_value_max = float(x_value.argmax())
        # x_value = np.arctanh(x_value)
        # x_value *= -10
        return x_policy, x_value_max


class MoveDictionary:
    def __init__(self):
        all_moves = self.generate_all_moves()
        self.move_index_dict = {move: index for index, move in enumerate(all_moves)}
        self.index_move_dict = {index: move for index, move in enumerate(all_moves)}
        #return move_index_dict


    def get_all_legal_moves(self, fen):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)  # Get an iterator of legal moves and convert to a list
        moves = [move.uci() for move in legal_moves]
        return [self.move_index_dict[move] for move in moves]

    def generate_all_squares(self):
        files = 'abcdefgh'
        ranks = '12345678'
        return [f + r for f in files for r in ranks]

    def is_within_board(self, file, rank):
        return 'a' <= file <= 'h' and '1' <= rank <= '8'

    def move_in_direction(self, start_square, file_step, rank_step, steps=8):
        moves = []
        start_file, start_rank = start_square[0], start_square[1]
        for step in range(1, steps + 1):
            new_file = chr(ord(start_file) + file_step * step)
            new_rank = chr(ord(start_rank) + rank_step * step)
            if self.is_within_board(new_file, new_rank):
                moves.append(new_file + new_rank)
            else:
                break
        return moves

    def generate_fairy_moves(self, start_square):
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Rook-like moves
            (1, 1), (1, -1), (-1, 1), (-1, -1),  # Bishop-like moves
            (2, 1), (2, -1), (-2, 1), (-2, -1),  # Knight-like moves
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        moves = []
        for file_step, rank_step in directions:
            if abs(file_step) == 2 or abs(rank_step) == 2:  # Knight-like moves
                moves.extend(self.move_in_direction(start_square, file_step, rank_step, steps=1))
            else:
                moves.extend(self.move_in_direction(start_square, file_step, rank_step))
        return moves

    def generate_promotion_moves(self, start_square, end_square):
        promotion_pieces = ['b', 'n', 'r', 'q']
        return [start_square + end_square + piece for piece in promotion_pieces]

    def generate_all_moves(self):
        all_squares = self.generate_all_squares()
        all_moves = []

        for start_square in all_squares:
            fairy_moves = self.generate_fairy_moves(start_square)
            for end_square in fairy_moves:
                all_moves.append(start_square + end_square)
                # Add promotion moves for pawns
                if start_square[1] == '7' and end_square[1] == '8' and abs(int(ord(start_square[0]))-int(ord(end_square[0]))) <= 1:  # White pawn promotion
                    all_moves.extend(self.generate_promotion_moves(start_square, end_square))
                if start_square[1] == '2' and end_square[1] == '1' and abs(int(ord(start_square[0]))-int(ord(end_square[0]))) <= 1:  # Black pawn promotion
                    all_moves.extend(self.generate_promotion_moves(start_square, end_square))
        return all_moves

move_dict_obj = MoveDictionary()


@dataclass
class Run_Config():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    batch_size: int = 1
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.01 * max_lr
    warmup_steps: float = 0.1
    max_steps: float = HyperParamConfig.max_steps #[0.70, 0.75, 0.80]
    total_steps: int = 1 #2 epochs for bay optim

@dataclass
class Chess_Config():
    block_size: int = 64 # n_squares
    vocab_size: int = 13 # 1 empty square, 6 own pieces, 6 opponent pieces
    n_layer: int = HyperParamConfig.n_layer # [16, 24, 32]
    n_head: int = HyperParamConfig.n_head # [8, 16, 32]
    n_embd: int = HyperParamConfig.n_embd # [128, 256, 512]]
    n_blocks_policyhead: int = HyperParamConfig.n_blocks_policyhead # [2,3,4]
    n_blocks_valuehead: int = HyperParamConfig.n_blocks_valuehead # [3,4,5]
    dropout: float = HyperParamConfig.dropout # [ 0.2, 0.3, 0.4, 0.5]


#torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chess(Chess_Config())
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
model = torch.compile(model)
model.eval()


def one_hot_vector(position, turn):
    if turn == 'b':
        position = position[::-1].swapcase()
    board_rows = position.split('/')
    x = []
    enc = {' ': 0, 'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6, 'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11,
           'K': 12}
    for board_row in board_rows:
        for i, square in enumerate(board_row):
            if square.isdigit():
                for j in range(int(square)):
                    x.append(0)
            else:
                x.append(enc[square])
    return x


with torch.no_grad():
    #fen = "6k1/pp2rpp1/4rb1p/P1p5/8/1K1P1N2/1P3P1P/2B4R w - - 0 1"
    fen = "r1bqkbnr/pppppppp/8/8/1n4Q1/4P3/PPPP1PPP/RNB1KBNR w KQkq - 0 1"
    board = chess.Board()
    print(board)
    while True:
        while True:
            try:
                player_move = input("What is your move? ")
                
                move = board.parse_san(player_move)
                
                if move in board.legal_moves:
                    board.push(move)
                    break
                else:
                    print("This move is not legal. Please try a different move.")
                
            except ValueError:
                # Handle cases where the input is not a valid SAN move
                print("Invalid move format. Please enter a valid chess move in standard algebraic notation.")
        full_fen = board.fen()
        position, turn, _, _, _, _ = full_fen.split()
        board_state = one_hot_vector(position, turn)
        data = torch.tensor(board_state)
        data = data.unsqueeze(dim=0)
        legal_indices = torch.tensor([move_dict_obj.move_index_dict[move.uci()] for move in list(board.legal_moves)])
        legal_indices = legal_indices.unsqueeze(dim=0)
        
        data, legal_indices = data.to(device), legal_indices.to(device)

        # Evaluate the loss
        x_policy, x_value = model(data, legal_indices)
        
        max_index = x_policy.argmax(dim=1).item()
        uci_move = move_dict_obj.index_move_dict[max_index]
        move = chess.Move.from_uci(uci_move)

        # Check if the move is legal
        if move in board.legal_moves:
            # Convert the move to SAN for display
            san_move = board.san(move)
            
            # Display the move in SAN format
            print("Model suggests:", san_move)
            print("eval = ", x_value)
            
            # Push the move onto the board
            #board.push(move)
        else:
            print("Model suggested an illegal move:", uci_move)
        while True:
            try:
                player_move = input("What is computer move? ")
                
                move = board.parse_san(player_move)
                
                if move in board.legal_moves:
                    board.push(move)
                    break
                else:
                    print("This move is not legal. Please try a different move.")
                
            except ValueError:
                # Handle cases where the input is not a valid SAN move
                print("Invalid move format. Please enter a valid chess move in standard algebraic notation.")





