import os
import math
import time
import inspect
from dataclasses import dataclass
import chess
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
import json
from torch.distributed import init_process_group, destroy_process_group





torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#file destination paths
#localhost
# db_path = "small_pgn_test_7.db"
# #db_path = '/content/drive/My Drive/small_pgn_test_7.db'
# #db_path = '/content/drive/My Drive/TCEC_github_v1.db'
# checkpoint_path = "checkpoint.pkl" #for bayesian optim model
# model_path = "test.pth"
# log_path = "log.txt"
# best_model_path = "best_model.pth"
# #runpod
#db_path = '/workspace/model/databases/TCEC_small_pgn_test.db'
db_path = '/workspace/database/lichess_7mil.db'
model_dir = "/workspace/runs/lichess_test_run/iters"
best_model_path = "/workspace/runs/lichess_test_run/best_model.pth"
model_path = None #correct path set ltr if save == True
log_path = None #correct path set ltr if write == True
debug_path = "debug.txt"

iteration = 0
existing_model_path = "../../runs/lichess_test_run/iters/state_dict_v30.pth"
pretrained_data = (10000, 4096, "lichess_7mil.db", 30) #(steps_completed, batch_size, db, iteration no of model we are loading)

run_training = True
run_validation = True
run_testing = False
write = True
save = True
actual_lr = 1e-3

with open("/workspace/runs/lichess_test_run/completed_indices.txt", "a+") as file:
    file.seek(0)
    completed_indices = file.read()
    completed_indices = completed_indices.split(",")[:-1]
    for i, value in enumerate(completed_indices):
        completed_indices[i] = int(value)
    while True:
        if iteration in completed_indices:
            iteration += 1
        else:
            break
    if write:
        file.write(f"{iteration},")
print(f"{iteration=}")


train_steps = 10000
n_limit = None
n_workers = 18
n1 = 0.8
n2 = 0.1
gpu_batch_size = 4096

#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int = 4096
    adamw_weight_decay: float = 1e-3
    gradient_clipping: float = 1.0
    max_lr: float = 1e-5
    max_steps: float = 0.80
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 128
    n_blocks_policyhead: int = 3
    n_blocks_valuehead: int = 4
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        assert model_config.n_embd % model_config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(model_config.n_embd, 3 * model_config.n_embd)
        # output projection
        self.c_proj = nn.Linear(model_config.n_embd, model_config.n_embd)
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
        

        self.fc = nn.Linear(model_config.n_embd, 1968)  # Fully connected layer to output a scalar


    def forward(self, x, masked_indices=None):
        #B, T, C = x.shape
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
            wpe = nn.Embedding(model_config.squares_size, model_config.n_embd),
            h = nn.ModuleList([Block(model_config) for _ in range(model_config.n_layer)]),
            ln_f = nn.LayerNorm(model_config.n_embd),
        ))

    def forward(self, board_state_tensor, special_tokens_tensor):
        # idx is of shape (B, T)
        B, T = board_state_tensor.size()
        assert T <= self.model_config.squares_size, f"Cannot forward sequence of length {T}, block size is only {self.model_config.squares_size}"
        # forward the token and position embeddings
        
        pos = torch.arange(0, T, dtype=torch.long, device=board_state_tensor.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(board_state_tensor) # token embeddings of shape (B, T, n_embd)
        special_emb = special_tokens_tensor.unsqueeze(-1).expand(B, self.model_config.special_size, self.model_config.n_embd)
        x = torch.cat((tok_emb + pos_emb, special_emb), dim=1)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        self.out = x
        return self.out
    



class Chess(nn.Module):

    def __init__(self, model_config, move_to_index, index_to_move):
        super().__init__()
        self.model_config = model_config
        self.transformer = Transformer(self.model_config)
        self.policy_head = PolicyHead(self.model_config)
        assert existing_model_path is not None
        print("loading existing state_dict")
        state_dict = torch.load(existing_model_path)
        adjusted_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()} #torch.compile is adding the prefix "_orig_mod." to all keys of the state_dict for some reason, need to remove it
        self.load_state_dict(adjusted_state_dict)
        self.move_to_index = move_to_index
        self.index_to_move = index_to_move

    def process_fen(self, fen):
        fen_parts = fen.split(" ")
        rows = fen_parts[0].split("/")
        turn = fen_parts[1]
        if fen_parts[1] == "b":
            
            rows = [row.swapcase() for row in rows][::-1]
            
            fen_parts[2] = fen_parts[2].swapcase()
        position = [0] #special token
        piece_dict = {" ":1, "p":2, "n":3, "b":4, "r":5, "q":6, "k":7, "P":8, "N":9, "B":10, "R":11, "Q":12, "K":13}
    
        for row in rows:
            for square in row:
                if square.isalpha():
                    position.append(piece_dict[square])
                else:
                    position.extend([1] * int(square))
        castling_rights = fen_parts[2]
        special_tokens = [0,0,0,0]
        for c in castling_rights:
            if c == "K":
                special_tokens[0] = 1
            elif c == "Q":
                special_tokens[1] = 1
            elif c == "k":
                special_tokens[2] = 1
            elif c == "q":
                special_tokens[3] = 1
        en_passant = fen_parts[3]
        if en_passant == "-":
            special_tokens.extend([0] * 9)
        else:
            file_index = ord(en_passant[0]) - 97
            special_tokens.extend([1] + [0] * file_index + [1] + [0] * (7 - file_index))
        

        board_state_tensor = torch.tensor(position).unsqueeze(dim=1)
        special_token_tensor = torch.tensor(special_tokens).unsqueeze(dim=1)

        return board_state_tensor, special_token_tensor, turn

    def reflect_uci_move(self, uci_move):
        out = ""
        out += uci_move[0]
        out += str(9 - int(uci_move[1]))
        out += uci_move[2]
        out += str(9 - int(uci_move[3]))
        return out

    def forward(self, fen):
        # idx is of shape (B, T)
        board_state_tensor, special_token_tensor, turn = self.process_fen(fen)
        x = self.transformer(board_state_tensor, special_token_tensor)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input)
        print(f"{x_policy.shape=}")
        move_index = x_policy.argmax()
        move = self.index_to_move[move_index]
        if turn == 'b':
            move = self.reflect_uci_move(move)
        return move








@dataclass
class Chess_Config():
    squares_size: int = 65 # n_squares + 1 for special token
    special_size: int = 13
    vocab_size: int = 27 # 1 special token, 1 empty square, 6 own pieces, 6 opponent pieces, 4 castling rights, 9 en_passant (1st for availabiltiy, other 8 to indicate file)
    n_layer: int = HyperParamConfig.n_layer # [16, 24, 32]
    n_head: int = HyperParamConfig.n_head # [8, 16, 32]
    n_embd: int = HyperParamConfig.n_embd # [128, 256, 512]]
    n_blocks_policyhead: int = HyperParamConfig.n_blocks_policyhead # [2,3,4]
    n_blocks_valuehead: int = HyperParamConfig.n_blocks_valuehead # [3,4,5]
    dropout: float = HyperParamConfig.dropout # [ 0.2, 0.3, 0.4, 0.5]

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chess(Chess_Config())
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
# If your policy head parameters are scattered in the model
policy_params = sum(p.numel() for name, p in model.named_parameters() if 'policy' in name and p.requires_grad)
print(f"Total number of parameters in the policy head: {policy_params}")

model = torch.compile(model)
model.eval()
fen = None
current_board = chess.Board(fen)
player_colour = input("player (w/b): ")
if player_colour == 'w':
    turn = 0
else:
    turn = 1

def get_player_move(player):
    while True:
        move_san = input(f"{type}'s move: ")
        try:
            move = current_board.parse_san(move_san)
            if move in current_board.legal_moves:
                return move
            else:
                print("move is not legal. try again.")
        except ValueError:
            print(f"Move {move_san} is invalid or not possible in the current board state.")


def fetch_engine_move():
    with torch.no_grad():
        move_uci_string = model(fen)
    move_uci = chess.Move.from_uci(move_uci_string)
    if move_uci not in current_board.legal_moves:
        print(f"computer suggested illegal move: {move_uci_string}")
    else:
        move_san = current_board.san(move_uci)
        print(f"computer suggested move: {move_san}")
    move_san = input("Enter computer's move: ")
    try:
        # Convert the SAN string to a Move object
        move = current_board.parse_san(move_san)

        # Check if the move is legal (parse_san only parses legal moves)
        if move in current_board.legal_moves:
            # Push the move to the board
            current_board.push(move)

            print(f"Move {move_san} is legal and has been played.")
            print(current_board)
        else:
            print(f"Move {move_san} is illegal.")
    except ValueError:
        print(f"Move {move_san} is invalid or not possible in the current board state.")
    print("computer suggests move")
    return move_san

def push_player_move():
    move_san = input("what is your move: ")

    print(current_board)
    uci_move = model(fen)
prev_board = None
prev_board_2 = None
while True:
    if turn == 0:
        move = get_player_move("player")
        if move == "r":
            current_board = prev_board_2.copy()
            continue
        current_board.push(move)
        turn = 1
    else:
        move = fetch_engine_move()
        current_board.push(move)






