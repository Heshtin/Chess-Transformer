import os
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
import joblib
import json
from torch.distributed import init_process_group, destroy_process_group
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

from blocks import CausalSelfAttention, MLP, Block, PolicyHead, Transformer

torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

existing_model_path = None

class Chess(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
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


    def forward(self, board_state_tensor, special_token_tensor, target_p_tensor=None, train_type="normal", legal_moves_tensor=None):
        # idx is of shape (B, T)
        x = self.transformer(board_state_tensor, special_token_tensor)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input, forward_pass="first", masked_indices=legal_moves_tensor) # (B, 1968)
        
        loss_p = None
        if target_p_tensor is not None:
            #target_p_tensor = target_p_tensor.long()
            loss_p = F.cross_entropy(x_policy, target_p_tensor)
        if train_type == "reevaluation":
            top_values, top_indices = torch.topk(x_policy, self.model_config.n_moves_reevaluate, dim=1)
            # Stack the top values and indices along a new dimension to create the desired shape (B, n, 2)
            reevaluation_moves_tensor = torch.stack((top_values, top_indices), dim=-1)
            reevaluation_indices = reevaluation_moves_tensor[:, :, 1]  # Extract the indices (shape (B, 3))
            # Step 2: Check if any move index matches the correct move index
            matches = (reevaluation_indices == target_p_tensor.unsqueeze(1))  # (B, 3) match matrix
            # Step 3: If there is a match, get the index of the matching move (0, 1, or 2)
            has_match = matches.any(dim=1)  # (B,) bool tensor indicating if a match exists
            matching_index = torch.argmax(matches, dim=1)  # (B,) index of the first matching move within the set of 3
            # Step 4: If no match, find the index of the move with the highest probability within the set of 3
            reevaluation_probs = reevaluation_moves_tensor[:, :, 0]  # Extract the probabilities (shape (B, 3))
            highest_prob_move_index = torch.argmax(reevaluation_probs, dim=1)  # (B,) index of max probability move within the set of 3
            # Step 5: Use the matching move index if a match exists; otherwise, use the highest probability move index
            final_target_index = torch.where(has_match, matching_index, highest_prob_move_index)
            x = self.transformer(board_state_tensor, special_token_tensor, reevaluation_moves_tensor)
            policy_input = x[:, 0:1, :].squeeze()
            x_policy = self.policy_head(policy_input, forward_pass="reevaluation") # (B, 1968)
        else:
            final_target_index = None
        loss_rp = None
        if final_target_index is not None:
            final_target_index = final_target_index.long()
            loss_rp = F.cross_entropy(x_policy, final_target_index)
        return x_policy, loss_p, loss_rp

    
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
