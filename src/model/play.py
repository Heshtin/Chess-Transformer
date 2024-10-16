import os
import math
import time
import inspect
import chess
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

from chess_model import Chess
from dataloader import ChessIterableDataset, pad_collate
from dataclass import VariableRunConfig, DataConfig, HyperParamConfig, RunConfig, ChessConfig
from auxilliary import retrieve_iteration_number, write_to_hyperparam
from uci_move_dict import MoveDictionary
    
move_dict_obj = MoveDictionary()
move_to_index = move_dict_obj.move_index_dict
index_to_move = move_dict_obj.index_move_dict


torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chess(ChessConfig(), index_to_move, device)
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
# If your policy head parameters are scattered in the model
policy_params = sum(p.numel() for name, p in model.named_parameters() if 'policy' in name and p.requires_grad)
print(f"Total number of parameters in the policy head: {policy_params}")

model = torch.compile(model)



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
