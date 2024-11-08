import os
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import chess

from blocks import CausalSelfAttention, MLP, Block, PolicyHead, Transformer

torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

existing_model_path = None

class ChessA(nn.Module):

    def __init__(self, model_config, device):
        super().__init__()
        self.model_config = model_config
        self.device = device
        self.transformer = Transformer(self.model_config)
        self.policy_head = PolicyHead(self.model_config)
        if existing_model_path is not None:
            print("loading existing state_dict")
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
            if module.out_features == 1968 or module.out_features == 3:
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

    def preprocess_input(board_state_tensor, special_tokens_tensor, encoding_scheme):
        # Check if we need to flip and swap colors based on the encoding scheme
        if encoding_scheme % 2 == 0:
            # Flip vertically and color swap for the board state
            board_state_tensor = board_state_tensor.view(-1, 8, 8).flip(1).reshape(-1, 64)

            # Color swap for piece codes
            black_mask = (board_state_tensor > 1) & (board_state_tensor < 8)
            white_mask = (board_state_tensor > 7) & (board_state_tensor < 14)
            board_state_tensor[black_mask] += 6
            board_state_tensor[white_mask] -= 6

            # Adjust special_tokens for castling rights due to color swap
            # Swapping index 0 with index 2, and index 1 with index 3
            temp_0, temp_1 = special_tokens_tensor[:, 0].clone(), special_tokens_tensor[:, 1].clone()
            special_tokens_tensor[:, 0], special_tokens_tensor[:, 2] = special_tokens_tensor[:, 2], temp_0
            special_tokens_tensor[:, 1], special_tokens_tensor[:, 3] = special_tokens_tensor[:, 3], temp_1

        return board_state_tensor, special_tokens_tensor


    def forward(self, board_state_tensor, special_token_tensor, legal_moves_tensor=None):
        # idx is of shape (B, T)
        board_state_tensor, special_token_tensor = self.preprocess_input(board_state_tensor, special_token_tensor, self.model_config.token_encoding_scheme)
        x = self.transformer(board_state_tensor, special_token_tensor, self.device)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input, masked_indices=legal_moves_tensor) # (B, 1968)
        return x_policy

    def play(self, fen):
        board_state_tensor, special_token_tensor = self.fen_to_vector(fen)
        return self.forward(board_state_tensor, special_token_tensor).argmax()
        
    
    def accuracy(self, board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor=None):
        x_policy = self.forward(board_state_tensor, special_token_tensor, legal_moves_tensor)
        top_index_tensor = torch.argmax(x_policy, dim=1)
        return (top_index_tensor == target_p_tensor).sum()
    
    def check_puzzle_batch(self, moves_tensor, puzzle_starting_fen, gpu_batch_size,puzzle_size):
        board = [chess.board(puzzle_starting_fen)] * gpu_batch_size
        board_state_tensor, special_token_tensor = torch.stack(self.fen_to_vector(puzzle_starting_fen))
        for i, move in enumerate(moves_tensor.tolist()):
            if i % 2 == 0:
                model_prediction = torch.argmax(self.forward(board_state_tensor, special_token_tensor), dim=1)
                if model_prediction != move:
                    return False
                board.push(move)
            else:
                board.push(move)
                board_state_tensor, special_token_tensor = self.fen_to_model_input(board.fen())
        return True
    
    def check_puzzle(self, moves_tensor, puzzle_starting_fen):
        board = chess.board(puzzle_starting_fen)
        board_state_tensor, special_token_tensor = self.fen_to_vector(puzzle_starting_fen)
        for i, move in enumerate(moves_tensor.tolist()):
            if i % 2 == 0:
                model_prediction = torch.argmax(self.forward(board_state_tensor, special_token_tensor), dim=1)
                if model_prediction != move:
                    return False
                board.push(move)
            else:
                board.push(move)
                board_state_tensor, special_token_tensor = self.fen_to_model_input(board.fen())
        return True
    
    def fen_to_position_tokens(self, fen):
        fen_parts = fen.split(" ")
        rows = fen_parts[0].split("/")
        if fen_parts[1] == "b":
            if self.colour_swap:
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

        return torch.tensor(position)
    
    def fen_to_special_tokens(self, fen):
        fen_parts = fen.split(" ")
        castling_rights = fen_parts[2]
        special_tokens = [1 if c in "KQkq" else 0 for c in castling_rights]
        en_passant = fen_parts[3]
        if en_passant == "-":
            special_tokens.extend([0] * 9)
        else:
            special_tokens.append(1)
            file_index = ord(en_passant[0]) - 97
            special_tokens.extend([0] * file_index)
            special_tokens.append(1)
            special_tokens.extend([0] * (7 - file_index))
        
        return torch.tensor(special_tokens)




        

    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Separate parameters for weight decay (>= 2 dim) and no weight decay (< 2 dim)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # Policy head final layer and other layer-specific learning rate adjustments
        policy_head_fc_params = list(self.policy_head.fc2.parameters()) #intentionally changed to fc2
        policy_head_fc_param_ids = {id(param) for param in policy_head_fc_params}
        
        policy_head_params = [param for name, param in self.named_parameters() 
                            if 'policy_head' in name and id(param) not in policy_head_fc_param_ids]
        
        rest_of_model_params = [param for name, param in self.named_parameters() 
                                if 'policy_head' not in name]

        # Combine weight decay and layer-specific learning rates
        optim_groups = [
            {'params': [p for p in decay_params if id(p) in policy_head_fc_param_ids], 'weight_decay': weight_decay, 'lr_type': -1},  # Final layer with weight decay
            {'params': [p for p in no_decay_params if id(p) in policy_head_fc_param_ids], 'weight_decay': 0.0, 'lr_type': -1},  # Final layer without weight decay

            {'params': [p for p in decay_params if id(p) in {id(param) for param in policy_head_params}], 'weight_decay': weight_decay, 'lr_type': -2},  # Policy head with weight decay
            {'params': [p for p in no_decay_params if id(p) in {id(param) for param in policy_head_params}], 'weight_decay': 0.0, 'lr_type': -2},  # Policy head without weight decay

            {'params': [p for p in decay_params if id(p) in {id(param) for param in rest_of_model_params}], 'weight_decay': weight_decay, 'lr_type': 1},  # Rest of model with weight decay
            {'params': [p for p in no_decay_params if id(p) in {id(param) for param in rest_of_model_params}], 'weight_decay': 0.0, 'lr_type': 1}  # Rest of model without weight decay
        ]
            
        # Optionally use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using adamW fused: {use_fused}")
        
        # Create the optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=use_fused)
        
        return optimizer
