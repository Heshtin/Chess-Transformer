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
db_path = '/workspace/database/lichess_2/combined_database.db'
model_dir = "/workspace/runs/lichess_run/iters"
best_model_path = "/workspace/runs/lichess_run/best_model.pth"
model_path = None #correct path set ltr if save == True
log_path = None #correct path set ltr if write == True
debug_path = "debug.txt"

iteration = 0
existing_model_path = "../runs/lichess_run/iters/state_dict_v68.pth"
pretrained_data = (10000, 4096, "combined_database.db", 30) #(steps_completed, batch_size, db, iteration no of model we are loading)

global_top_k_prob_indices = 1

run_training = False
run_validation = True
run_testing = False
write = False
save = False


global_train_type = "normal"
global_masking = False
# if global_masking:
#     torch._dynamo.config.suppress_errors = True
train_steps = 50000
global_n_limit = None
n_workers = 1
global_n1 = 0.9
global_n2 = 0.1
gpu_batch_size = 1024



#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int = 1024
    adamw_weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    max_lr: float = 5e-3
    max_steps: float = 0.80
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
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
 
        
        if reevaluation_moves_tensor is not None:
            moves_tensor = reevaluation_moves_tensor[:, :, 0].long()  # (B, n_moves_reevaluate)
            prob_tensor = reevaluation_moves_tensor[:, :, 1].float()  # (B, n_moves_reevaluate)
            raw_move_emb = self.transformer.me(moves_tensor)  # (B, n_moves_reevaluate, n_embd)
            #s = torch.tensor([0], dtype=torch.int64)
            scaling_emb = prob_tensor.unsqueeze(-1) * self.se # self.transformer.se(torch.tensor([0], dtype=torch.int64))  # (B, n_moves_reevaluate, n_embd)
            rank_indices = torch.arange(0, 3, device=device).unsqueeze(0).expand(B, -1)  # Shape: (B, 3)
            rank_emb = self.transformer.mre(rank_indices)  # rank embeddings
            move_emb = raw_move_emb + scaling_emb + rank_emb

        else:
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


    def forward(self, top_k_prob_indices, board_state_tensor, special_token_tensor, target_p_tensor=None, train_type="normal", legal_moves_tensor=None):
        # idx is of shape (B, T)
        x = self.transformer(board_state_tensor, special_token_tensor)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input) # (B, 1968)
        
        _, final_target_indices = torch.topk(x_policy, top_k_prob_indices, dim=1) # (B,3)
        matches = (target_p_tensor.unsqueeze(1) == final_target_indices)
        num_matches = matches.any(dim=1).sum().item()
        return num_matches
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Separate parameters for weight decay (>= 2 dim) and no weight decay (< 2 dim)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # Policy head final layer and other layer-specific learning rate adjustments
        policy_head_fc_params = list(self.policy_head.fc.parameters())
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

   
    
class ChessIterableDataset(IterableDataset):
    def __init__(self, db_path, split, n_limit, n1=0.8, n2=0.1, masking=False):
        self.db_path = db_path
        self.split = split
        self.n1 = n1  # Proportion of training data
        self.n2 = n2  # Proportion of validation data
        self.n_limit = n_limit  # Optional limit on the number of data points
        self.masking = masking
        self.masking_query = ", legal_moves" if masking else ""

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate limits for each split
        total_query = "SELECT COUNT(*) FROM chess_analysis;"
        cursor.execute(total_query)
        total_rows = cursor.fetchone()[0]
        
        if self.n_limit is not None:
            total_rows = min(total_rows, self.n_limit)  # Adjust total_rows based on n_limit

        train_limit = int(total_rows * self.n1)
        val_limit = int(total_rows * self.n2)
        test_limit = total_rows - train_limit - val_limit

        # Prepare the query based on the split
        if self.split == 'train':
            query = f"SELECT board_state, special_tokens, next_move, turn{self.masking_query} FROM chess_analysis LIMIT {train_limit};"
        elif self.split == 'val':
            query = f"SELECT board_state, special_tokens, next_move, turn{self.masking_query} FROM chess_analysis LIMIT {val_limit} OFFSET {train_limit};"
        elif self.split == 'test':
            query = f"SELECT board_state, special_tokens, next_move, turn{self.masking_query} FROM chess_analysis LIMIT {test_limit} OFFSET {train_limit + val_limit};"

        cursor.execute(query)
        for row in cursor.fetchall():
            board_state = json.loads(row[0])
            for i, value in enumerate(board_state):
                if 1 < value < 8: #0,1 for CLS, empty sq
                    board_state[i] += 6
                elif value >= 8:
                     board_state[i] -= 6
            # board_state = []
            # for i in range(7, -1, -1):
            #     board_state.extend(pre_board_state[i*8:(i+1)*8])
            turn = row[3]
            turn_token = 0 if turn == "w" else 1
            special_tokens = json.loads(row[1])
            # special_tokens = [turn_token] + special_tokens
            # special_tokens[0], special_tokens[2] = special_tokens[2], special_tokens[0]
            # special_tokens[1], special_tokens[3] = special_tokens[3], special_tokens[1]
            target_move_index = row[2]
            board_state_tensor, special_token_tensor, target_move_tensor = torch.tensor(board_state, dtype=torch.int64), torch.tensor(special_tokens, dtype=torch.int64), torch.tensor(target_move_index, dtype=torch.int64)
            if self.masking:
                legal_moves_list = json.loads(row[3])
                yield (board_state_tensor, 
                    special_token_tensor,
                    target_move_tensor,
                    legal_moves_list) #returned as list initially to allow for padding
            else:
                yield (board_state_tensor, 
                    special_token_tensor,
                    target_move_tensor)
        
        conn.close()

def pad_collate(batch):
    # Unpack the batch into respective tensors and lists
    board_states = torch.stack([data[0] for data in batch])       # Already tensors, just stack them
    special_tokens = torch.stack([data[1] for data in batch])      # Already tensors, just stack them
    target_moves = torch.stack([data[2] for data in batch])        # Already tensors, just stack them

    # Handle legal_moves if masking is enabled (these are lists, need padding)
    if len(batch[0]) == 4:  # Check if legal_moves are present
        legal_moves = [torch.tensor(data[3], dtype=torch.int64) for data in batch]
        
        # Pad legal_moves to the same length automatically
        legal_moves_padded = pad_sequence(legal_moves, batch_first=True, padding_value=-1) #pad_sequence is an imported function
    else:
        legal_moves_padded = None

    return board_states, special_tokens, target_moves, legal_moves_padded


train_loader = None
val_loader = None

#indv batch size always 16 (or as much as GPU can handle)
if run_training:
    train_dataset = ChessIterableDataset(db_path, 'train', global_n_limit, global_n1, global_n2, global_masking)
    train_loader = DataLoader(train_dataset, batch_size=gpu_batch_size, num_workers=n_workers, collate_fn=pad_collate)
if run_validation:
    val_dataset = ChessIterableDataset(db_path, 'val', global_n_limit, global_n1, global_n2, global_masking)
    val_loader = DataLoader(val_dataset, batch_size=gpu_batch_size, num_workers=n_workers, collate_fn=pad_collate)
if run_testing:
    test_dataset = ChessIterableDataset(db_path, 'test', global_n_limit, global_n1, global_n2, global_masking)
    test_loader = DataLoader(test_dataset, batch_size=gpu_batch_size, num_workers=n_workers, collate_fn=pad_collate)

@dataclass
class run_config():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    batch_size: int = gpu_batch_size
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.01 * max_lr
    warmup_steps: float = 0.0075
    max_steps: float = HyperParamConfig.max_steps #[0.70, 0.75, 0.80]
    total_steps: int = train_steps #2 epochs for bay optim

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
    n_possible_moves: int = 1968 #no of uci moves (moves in output)
    n_moves_reevaluate: int = 3 #no of moves that are reevaluated in 2nd forward pass

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




run_config = run_config()





total_batch_size = run_config.total_batch_size # used for alphazero
batch_size = run_config.batch_size
assert total_batch_size % batch_size == 0
grad_accum_steps = total_batch_size // batch_size



def validation(model, val_loader, device, run_config, log_path):
    model.eval()
    val_iter = iter(val_loader)
    print("starting validation")
    accuracy_count = 0.0
    total_count = 0.0
    with torch.no_grad():
        step = 0
        while True:
            try:
                board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor = next(val_iter)
            except StopIteration:
                break
            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)
            if legal_moves_tensor:
                legal_moves_tensor.to(device)
            # Evaluate the loss
            n_matches = model(global_top_k_prob_indices, board_state_tensor, special_token_tensor, target_p_tensor)
            print(f"step={step}, n_matches={n_matches}")
            accuracy_count += n_matches
            total_count += gpu_batch_size
            step += 1
            if step == 1000:
                break
        
    print(f"Validation accuracy: | accuracy={accuracy_count/total_count} accuracy_count={accuracy_count} | total_count={total_count}")

if run_validation:
    validation(model, val_loader, device, run_config, log_path)


# if __name__ == '__main__':
#     main()
