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
import json
from torch.distributed import init_process_group, destroy_process_group





torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
existing_model_path = "../runs/lichess_run/iters/state_dict_v15.pth"


#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int = 4096
    adamw_weight_decay: float = 1e-3
    gradient_clipping: float = 1.0
    max_lr: float = 1e-3
    max_steps: float = 0.80
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
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

    def __init__(self, model_config, index_to_move, device):
        super().__init__()
        self.model_config = model_config
        self.index_to_move = index_to_move
        self.device = device
        self.transformer = Transformer(self.model_config)
        self.policy_head = PolicyHead(self.model_config)
        assert existing_model_path is not None; "please input existing model path"
        print("loading existing state_dict")
        state_dict = torch.load(existing_model_path)
        adjusted_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()} #torch.compile is adding the prefix "_orig_mod." to all keys of the state_dict for some reason, need to remove it
        self.load_state_dict(adjusted_state_dict)
        


    def forward(self, fen, move_rank):
        # idx is of shape (B, T)
        board_state_tensor, special_token_tensor, turn = self.fen_to_vector(fen)
        board_state_tensor, special_token_tensor = board_state_tensor.to(self.device), special_token_tensor.to(self.device)
        x = self.transformer(board_state_tensor, special_token_tensor)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input)
        top_n_values, indices = torch.topk(x_policy, move_rank)
        max_index = indices[-1].item()
        # print(f"{top_n_values=}")
        # print(f"{max_index=}")
        # import sys; sys.exit(0)
        uci_move = self.index_to_move[max_index]
        if turn == "b":
            uci_move = self.flip_uci(uci_move)
        

        return uci_move

    def flip_uci(self, uci_move_string):
        out = ""
        out+=uci_move_string[0]
        out+=str(9 - int(uci_move_string[1]))
        out+=uci_move_string[2]
        out+=str(9 - int(uci_move_string[3]))
        if len(uci_move_string) == 5:
            out+=uci_move_string[4]
        return out

    def fen_to_vector(self, fen):
        fen_parts = fen.split(" ")
        rows = fen_parts[0].split("/")
        turn = fen_parts[1]
        if fen_parts[1] == "b":
            
            rows = [row.swapcase() for row in rows][::-1]
            
            fen_parts[2] = fen_parts[2].swapcase()
            #fen_parts[1] = 1
        # else:
        #     #fen_parts[1] = 0
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
        
        board_state_tensor = torch.tensor(position).unsqueeze(dim=0)
        special_token_tensor = torch.tensor(special_tokens).unsqueeze(dim=0)


        return board_state_tensor, special_token_tensor, turn





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
move_to_index = move_dict_obj.move_index_dict
index_to_move = move_dict_obj.index_move_dict










@dataclass
class run_config():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    batch_size: int = 0
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.01 * max_lr
    warmup_steps: float = 0.01
    max_steps: float = HyperParamConfig.max_steps #[0.70, 0.75, 0.80]
    total_steps: int = 0 #2 epochs for bay optim

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
model = Chess(Chess_Config(), index_to_move, device)
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
# If your policy head parameters are scattered in the model
policy_params = sum(p.numel() for name, p in model.named_parameters() if 'policy' in name and p.requires_grad)
print(f"Total number of parameters in the policy head: {policy_params}")

model = torch.compile(model)



run_config = run_config()

starting_fen = None #"6k1/3p4/3N2pp/5p2/8/5P2/6PP/6rK w - - 0 1" #"3rk2r/1p2q3/p1p1Pppp/5R2/2P5/4Q3/P1P3PP/4R1K1 w k - 0 1"

board = chess.Board(starting_fen) if starting_fen else chess.Board()
    
player_color = ''
while player_color.lower() not in ['white', 'black']:
    player_color = input("Choose your color (white/black): ").strip()

if player_color.lower() == 'white':
    player_turn = chess.WHITE
else:
    player_turn = chess.BLACK

game_over = False
while not game_over:
    print("\nCurrent Board:")
    print(board)
    print()
    
    if board.turn == player_turn:
        # Player's turn
        move_made = False
        while not move_made:
            move_input = input("Your move (or type 'back' to go back a move): ").strip()
            if move_input.lower() == 'back':
                # Go back a move
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print("Went back one full move.")
                else:
                    print("Cannot go back any further.")
                continue
            try:
                # Try parsing as SAN notation
                move = board.parse_san(move_input)
            except ValueError:
                try:
                    # Try parsing as UCI notation
                    move = chess.Move.from_uci(move_input)
                except ValueError:
                    print("Invalid move format. Use SAN (e.g., Nf3) or UCI notation (e.g., g1f3).")
                    continue
            if move in board.legal_moves:
                board.push(move)
                move_made = True
            else:
                print("Illegal move. Try again.")
    else:
        # Computer's turn
        fen = board.fen()
        move_rank = 1
        suggested_move_uci = model(fen, move_rank)
        move = chess.Move.from_uci(suggested_move_uci)
        san_move = board.san(move)
        print(f"Computer's suggested move: {san_move} ({suggested_move_uci})")
        
        move_made = False
        while not move_made:
            move_input = input("Enter the move for the computer (or type 'back' to go back a move): ").strip()
            if move_input.lower() == 'back':
                # Go back a move
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print("Went back one full move.")
                else:
                    print("Cannot go back any further.")
                continue
            elif move_input.lower() == 'n':
                move_rank += 1
                suggested_move_uci = model(fen, move_rank)
                move = chess.Move.from_uci(suggested_move_uci)
                san_move = board.san(move)
                print(f"Computer's {move_rank}th suggested move: {san_move} ({suggested_move_uci})")
                continue

            try:
                # Try parsing as SAN notation
                move = board.parse_san(move_input)
            except ValueError:
                try:
                    # Try parsing as UCI notation
                    move = chess.Move.from_uci(move_input)
                except ValueError:
                    print("Invalid move format. Use SAN (e.g., Nf3) or UCI notation (e.g., g1f3).")
                    continue
            if move in board.legal_moves:
                board.push(move)
                move_made = True
            else:
                print("Illegal move. Try again.")
    
    # Check for game over conditions
    if board.is_checkmate():
        print(board)
        if board.turn == player_turn:
            print("Checkmate! You lose.")
        else:
            print("Checkmate! You win.")
        game_over = True
    elif board.is_stalemate():
        print(board)
        print("Stalemate! It's a draw.")
        game_over = True
    elif board.is_insufficient_material():
        print(board)
        print("Draw due to insufficient material.")
        game_over = True
    elif board.can_claim_threefold_repetition():
        print(board)
        print("Draw by threefold repetition.")
        game_over = True
