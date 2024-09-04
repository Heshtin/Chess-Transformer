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
from concurrent.futures import ThreadPoolExecutor
import threading



model_path = "../runs/bay_optim_run_1/iters2/state_dict_v71.pth"
import graphviz

def visualize_mcts_tree(root_node):
    dot = graphviz.Digraph()

    def add_node(dot, node, parent_id=None):
        node_id = str(id(node))
        label = f"Value: {node.total_value}\nVisits: {node.n_visits}\nMove: {node.reached_by_move}"
        dot.node(node_id, label)
        
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        
        for child in node.children:
            add_node(dot, child, node_id)

    add_node(dot, root_node)
    return dot




class Node():
    global_node_storage = []
    
    def __init__(self, board, model, device, move_to_index, index_to_move, prob, max_threads, reached_by_move=None):
        self.board = board
        self.total_value = 0
        self.n_visits = 0
        self.n_total_visits = 0
        self.c_puct = 1.0
        self.identical_nodes = []
        self.ucb = float("inf")
        self.children = []
        self.reached_by_move = reached_by_move
        self.max_rollout_depth = 10
        self.node_lock = threading.Lock()
        
        self.move_to_index = move_to_index
        self.index_to_move = index_to_move

        self.max_threads = max_threads
        

        self.prob = prob

        self.device = device
        self.model = model

        self.policy_vector = self.get_policy_value(self.board)[0]
        

        self.update_node_storage()

    
    # def __call__(self, time_limit):
    #     self.search(time_limit)
    #     ucb_max = float("-inf")
    #     max_child = self.children[0]
    #     for child in self.children:
    #         if child.n_visits != 0:
    #             if child.ucb > ucb_max:
    #                 ucb_max = child.ucb
    #                 max_child = child
    #     return max_child.reached_by_move, max_child

    def __call__(self, time_limit):
        self.search(time_limit)
        ucb_max = float("-inf")
        max_child = self.children[0]
        for child in self.children:
            if child.n_visits > max_child.n_visits:
                max_child = child
            elif child.n_visits == max_child.n_visits:
                if child.total_value > max_child.total_value:
                    max_child = child
        return max_child.reached_by_move, max_child

    def prune_tree(self, keep_node=None):
        if keep_node and keep_node in self.children:
            self.children.remove(keep_node)
        for child in self.children:
            child.prune_tree()
        self.children = []
        position = self.position_node_storage
        if self in Node.global_node_storage[position]:
            Node.global_node_storage[position].remove(self)
        # if len(Node.global_node_storage[position]) == 0: #if inner nested loop is empty delete it
        #     del Node.global_node_storage[position]
        #immediately del memory intensive attributes, garbage collector will clear the rest after unknown amount of time
        #del self.board
        #del self.policy_vector
        

    def update_board(self, board):
        self.board = board
        self.policy_vector = self.get_policy_value(self.board)[0]
        

    def search(self, time_limit):
        start_time = time.time()
        n_searches = 0
        with ThreadPoolExecutor() as executor:
            while time.time() - start_time < time_limit:
                print("executing")
                futures = [executor.submit(self.run) for _ in range(self.max_threads)]
                for future in futures:
                    future.result()
                n_searches += len(futures)
        print(f"{n_searches=}")
        



    def update_node_storage(self):
        for i, node_list in enumerate(Node.global_node_storage):
            if node_list and node_list[0].board == self.board: #if node_list to ensure list isnt empty (from beign pruned)
                Node.global_node_storage[i].append(self)
                self.position_node_storage = i
                return None
        Node.global_node_storage.append([self])
        self.position_node_storage = len(Node.global_node_storage) - 1
        


    def run(self):
        if self.n_visits == 0:
            value = self.rollout()


        elif not self.children:
            self.expand_node(self.policy_vector)
            prob_max = -1
            max_index = -1
            for i, child in enumerate(self.children):
                if child.prob > prob_max:
                    max_index = i
                    prob_max = child.prob
            value = self.children[max_index].run()
        

        else:
            ucb_max = float("-inf")
            inf_indices = []
            max_index = -1
            for i, child in enumerate(self.children):
                if child.ucb == float("inf"):
                    inf_indices.append(i)
                elif child.ucb > ucb_max:
                    max_index = i
                    ucb_max = child.ucb
            if len(inf_indices) > 0:
                prob_max = -1
                for i in inf_indices:
                    child = self.children[i]
                    if child.prob > prob_max:
                        max_index = i
                        prob_max = child.prob
                
            value = self.children[max_index].run()
        

        for node in Node.global_node_storage[self.position_node_storage]:
            with node.node_lock:
                node.n_total_visits += 1

        with self.node_lock:
            self.total_value += value
            self.n_visits += 1
            self.update_ucb()

        return value

    def rollout(self):
        board_copy = self.board.copy()
        for i in range(self.max_rollout_depth):
            if self.evaluate_terminal_state(board_copy) is not None:
                return self.evaluate_terminal_state(board_copy)
            current_policy = self.get_policy_value(board_copy)[0]
            max_index = current_policy.argmax()
            uci_move = self.index_to_move[max_index.item()]
            move = chess.Move.from_uci(uci_move)
            board_copy.push(move)
        return self.get_policy_value(board_copy)[1] #return the value output by model after rollout depth reached

    def evaluate_terminal_state(self, board):
        if board.is_checkmate():
            return 1 if board.turn == chess.BLACK else -1 # If the game is checkmate, the winner is the opponent of the player to move
        elif board.is_stalemate() or board.is_fifty_moves() or board.is_repetition(3) or board.is_insufficient_material():
            return 0
        
        return None


    
    def update_ucb(self):
        self.ucb = (self.total_value / self.n_visits) + self.c_puct * self.prob * (math.sqrt(self.n_total_visits) / (1 + self.n_visits))

    
    def expand_node(self, policy_vector):
        moves=[]
        for i, value in enumerate(policy_vector[0]):
            if value != 0.0:
                move = self.index_to_move[i]
                moves.append(move)
                uci_move = self.index_to_move[i]
                move = chess.Move.from_uci(uci_move)
                board_copy = self.board.copy()
                board_copy.push(move)

                self.children.append(Node(board_copy, self.model, self.device, self.move_to_index, self.index_to_move, value, 1, reached_by_move=uci_move))



    

    def get_policy_value(self, board):
        full_fen = board.fen()
        position, turn, _, _, _, _ = full_fen.split()
        board_state = self.one_hot_vector(position, turn)
        data = torch.tensor(board_state)
        data = data.unsqueeze(dim=0)
        legal_indices = torch.tensor([move_dict_obj.move_index_dict[move.uci()] for move in list(board.legal_moves)])
        legal_indices = legal_indices.unsqueeze(dim=0)
        data, legal_indices = data.to(self.device), legal_indices.to(self.device)
        x_policy, x_value = self.model(data, legal_indices)
        return x_policy, x_value
    
    def one_hot_vector(self, position, turn):
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
    





@dataclass
class HyperParamConfig:
    total_batch_size: int = 4096
    adamw_weight_decay: float = 1e-3
    gradient_clipping: float = 1.0
    max_lr: float = 1e-2
    max_steps: float = 0.80
    n_layer: int = 16
    n_head: int = 8
    n_embd: int = 128
    n_blocks_policyhead: int = 3
    n_blocks_valuehead: int = 4
    dropout: float = 0.5

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


class ConvBlock(nn.Module):
    def __init__(self, model_config, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(p=model_config.dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        #no residual connection bcos output size is different from input size
        return x

class ValueHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        inchannels = model_config.n_embd
        log_reduction = int(math.log2(model_config.n_embd))
        common_reduction = log_reduction // model_config.n_blocks_valuehead
        n_additional_reductions = log_reduction % model_config.n_blocks_valuehead
        self.blocks = nn.ModuleList()
        for i in range(model_config.n_blocks_valuehead):
            reduction = common_reduction if n_additional_reductions+i < model_config.n_blocks_valuehead else common_reduction + 1
            outchannels = inchannels // 2**(reduction)
            self.blocks.append(ConvBlock(model_config, in_channels=inchannels, out_channels=outchannels))
            inchannels = outchannels
        self.fc = nn.Linear(8 * 8, 1)
        self.output_tanh = nn.Tanh()

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)
        x = x.view(B, C, 8, 8)
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        x = self.fc(x)  # Fully connected layer to output a scalar
        x = self.output_tanh(x)
        return x





class PolicyHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        inchannels = model_config.n_embd
        log_reduction = int(math.log2(model_config.n_embd)) - 5
        common_reduction = log_reduction // model_config.n_blocks_policyhead
        n_additional_reductions = log_reduction % model_config.n_blocks_policyhead
        self.blocks = nn.ModuleList()
        for i in range(int(model_config.n_blocks_policyhead)):
            reduction = common_reduction if n_additional_reductions+i < model_config.n_blocks_policyhead else common_reduction + 1
            outchannels = inchannels // 2**(reduction)
            self.blocks.append(ConvBlock(model_config, in_channels=inchannels, out_channels=outchannels))
            inchannels = outchannels

        self.fc = nn.Linear(64 * 32, 1968)  # Fully connected layer to output a scalar


    def forward(self, x, masked_indices):
        B, T, C = x.shape
        x = x.transpose(1, 2)
        x = x.view(B, C, 8, 8)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(B, -1)
        x = self.fc(x)
        x = self.masked_softmax(x, masked_indices)
        # with open("policy_vector.txt", "w") as file:
        #     for b in x:
        #         file.write(str(b.tolist()))
        #         file.write(",")
        #     print("written")
        # with open("legal_indices.txt", "w") as file:
        #     for b in masked_indices:
        #         file.write(str(b.tolist()))
        #         file.write(",")
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
            print(b)
            print(f"{batch_tensor=}")
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
        print("loading existing state_dict")
        state_dict = torch.load(model_path)
        adjusted_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()} #torch.compile is adding the prefix "_orig_mod." to all keys of the state_dict for some reason, need to remove it
        self.load_state_dict(adjusted_state_dict)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            a = math.sqrt(1/2048)
            if module.out_features == 1968:  # Assuming this is the size of the policy vector
                # Initialize weights to small uniform values
                nn.init.uniform_(module.weight, -a, a)
                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif module.out_features == 1:  # Assuming this is the value output
                # Initialize weights to small uniform values
                nn.init.uniform_(module.weight, -a, a)
                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            else:
                nn.init.xavier_uniform_(module.weight)
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


    def forward(self, data, legal_indices):
        # idx is of shape (B, T)
        B, T = data.size()
        x = self.transformer(data)

        x_policy = self.policy_head(x, legal_indices)
        x_policy = F.softmax(x_policy, dim=1)
        x_value = self.value_head(x)
        x_value = x_value.squeeze().item()
        x_value = np.arctanh(x_value)
        
        return x_policy, x_value








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
class Run_Config():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    batch_size: int = 1
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.1 * max_lr
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
#model = torch.compile(model)
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
    #fen = "rnbqkbnr/pppppp1p/8/8/8/R2K3P/PPPPNp2/RNBQ1B2 w KQkq - 0 1"
    #fen = "rnbqkbnr/pppppp1p/8/8/4P3/R2K4/PPPPNp2/RNBQ1B2 b KQkq - 0 1"
    fen = "rnbqkbnr/pppppp2/8/7p/4P3/R2K3N/PPPP1p2/RNBQ1B2 w KQkq - 0 1"
    board = chess.Board(fen)
    root_node = Node(board, model, device, move_to_index, index_to_move, 1, 1) #fix so that there is no rollout at root node
    first_iteration = True
    i = 0
    while True:
        if first_iteration:
            uci_move, current_node = root_node(30.0)
            first_iteration = False
        else:
            uci_move, current_node = root_node(20.0)
        move = chess.Move.from_uci(uci_move)
        if move in board.legal_moves:
            san_move = board.san(move)
            print("Model suggests:", san_move)
            board.push(move)
        else:
            print("Model suggested an illegal move:", uci_move)
        # Assuming root_node is your MCTS tree root
        dot = visualize_mcts_tree(root_node)
        dot.render(f'mcts_tree_{i}', view=False)  # This will create and open a PDF of the tree
        root_node.prune_tree(current_node)
        dot = visualize_mcts_tree(root_node)
        dot.render(f'mcts_tree_pruned_{i}', view=False)
        root_node = current_node
        
        print(root_node.children)
        while True:
            try:
                player_move = input("What is your move? ")
                
                move = board.parse_san(player_move)
                
                if move in board.legal_moves:
                    board.push(move)
                    uci_move = move.uci()
                    for child in current_node.children:
                        if child.reached_by_move == uci_move:
                            current_node = child 
                    current_node.update_board(board)
                    break
                else:
                    print("This move is not legal. Please try a different move.")
                
            except ValueError:
                # Handle cases where the input is not a valid SAN move
                print("Invalid move format. Please enter a valid chess move in standard algebraic notation.")
        print(root_node.children)
        root_node.prune_tree(current_node)
        root_node = current_node
        i += 1












    