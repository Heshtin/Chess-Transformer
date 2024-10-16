import sys
import importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
sys.path.append('../models/model1')
import blocks
importlib.reload(chess_model)  # Reloads the module
from blocks import CausalSelfAttention, MLP, Block, PolicyHead, Transformer # Now import the class

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

    def forward(self, top_k_prob_indices, board_state_tensor, special_token_tensor, target_p_tensor=None, train_type="normal", legal_moves_tensor=None):
        # idx is of shape (B, T)
        x = self.transformer(board_state_tensor, special_token_tensor)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input) # (B, 1968)
        
        _, final_target_indices = torch.topk(x_policy, top_k_prob_indices, dim=1) # (B,3)
        matches = (target_p_tensor.unsqueeze(1) == final_target_indices)
        num_matches = matches.any(dim=1).sum().item()
        return num_matches
