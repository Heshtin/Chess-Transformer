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
            if module.out_features == 1968:
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


    def forward(self, board_state_tensor, special_token_tensor, target_p_tensor=None,):
        # idx is of shape (B, T)
        x = self.transformer(board_state_tensor, special_token_tensor)
        policy_input = x[:, 0:1, :].squeeze()
        x_policy = self.policy_head(policy_input)
        #print(f"{x_policy.shape}")
        #import sys; sys.exit(0)
        loss_p = None
        if target_p_tensor is not None:
            target_p_tensor = target_p_tensor.float()
            #x_policy = F.softmax(x_policy, dim=1)
            target_p_tensor = F.softmax(target_p_tensor, dim=1)
            # Apply log_softmax to the predicted probabilities
            x_policy_log = torch.log_softmax(x_policy, dim=1)

            # Calculate KL Divergence
            loss_p = F.kl_div(x_policy_log, target_p_tensor, reduction='batchmean')
            #loss_p = self.euclidean_loss(x_policy, target_p_tensor)
        return x_policy, loss_p

    def euclidean_loss(self, pred, target):
        return torch.sqrt(torch.sum(abs(pred - target), dim=1)).mean()
    def cross_entropy_loss(self, prob_pred, prob_target, eps=1e-8):
    # Ensure prob_pred is clamped to avoid log(0)
        prob_pred = torch.clamp(prob_pred, min=eps, max=1.0 - eps)  # Correct usage
        return -torch.sum(prob_target * torch.log(prob_pred), dim=1).mean()


    
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

    

class ChessDataset(Dataset):
    def __init__(self, db_path, n_limit, split, n1=0.8, n2=0.1, transform=None):
        self.transform = transform
        self.n_limit_query = f" LIMIT {n_limit};" if n_limit else ";"
        self.df = self.get_dataframe(db_path)
        self.n_total = len(self.df)
        self.n_train = int(n1 * self.n_total)
        self.n_val = int(n2 * self.n_total)
        self.n_test = self.n_total - self.n_train - self.n_val

        if split == 'train':
            self.start_row = 0
            self.end_row = self.n_train
        elif split == 'val':
            self.start_row = self.n_train
            self.end_row = self.n_train + self.n_val
        elif split == 'test':
            self.start_row = self.n_train + self.n_val
            self.end_row = self.n_total
        self.data = self.convert_df(self.start_row, self.end_row)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        board_state_tensor = self.data[0][idx]
        special_token_tensor = self.data[1][idx]
        target_p_tensor = self.data[2][idx]

        # if self.transform:
        #     x = self.transform(x)

        return board_state_tensor, special_token_tensor, target_p_tensor

    def get_dataframe(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        #query = f"SELECT move_index, legal_move_indices, x, evaluation FROM evaluations WHERE ABS(evaluation) > 1;"
        query = f"SELECT board_state, special_tokens, next_move, legal_moves, moves_left FROM chess_analysis{self.n_limit_query}" # WHERE n_pieces <= 12" #Change number of pieces per position
        cursor.execute(query)
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=['board_state', 'special_tokens', 'next_move', 'legal_moves', 'moves_left'])
        df = df.drop_duplicates(subset=['board_state', 'special_tokens', 'next_move', 'legal_moves', 'moves_left'], keep='first')
        print("fetched df")
        return df

    def convert_df(self, start_index, end_index):
        length = 0
        board_state_list, special_token_list, target_p_list = [], [], [] #squares, policy vector (output), castling/en-passant rights, legal_moves_indices
        #legal_moves_padding = self.df.iloc[start_index:end_index]['num_legal_moves'].max()
        for index, row in self.df.iloc[start_index:end_index].iterrows():
            board_state = json.loads(row['board_state'])
            special_tokens = json.loads(row['special_tokens'])
            target_move_index = row['next_move']
            legal_moves = json.loads(row['legal_moves'])
            moves_left = row['moves_left']
            target_p = [-10.0] * 1968
            for move_index in legal_moves:
                target_p[move_index] = 0.0
            if target_move_index in legal_moves:
                target_p[target_move_index] = 4.0 + (1.0 / moves_left)
            else:
                target_p[target_move_index] = -25.0
            board_state_list.append(board_state)
            if len(special_tokens) == 12:
                length += 1
                special_tokens.insert(4, 1)
            
            special_token_list.append(special_tokens)
            target_p_list.append(target_p)

            if index % 1000000 == 0:
                print(f"processed {index} rows")

        print("finished iterating through all rows")


        board_state_tensor = torch.tensor(board_state_list)
        print(f"{length=}")
        special_token_tensor = torch.tensor(special_token_list)
        target_p_tensor = torch.tensor(target_p_list)

        return board_state_tensor, special_token_tensor, target_p_tensor





#indv batch size always 16 (or as much as GPU can handle)
if run_training:
    train_dataset = ChessDataset(db_path, n_limit, 'train', n1=0.8, n2=0.1)
    train_loader = DataLoader(train_dataset, batch_size=gpu_batch_size, shuffle=True, num_workers=n_workers)
if run_validation:
    val_dataset = ChessDataset(db_path, n_limit, 'val', n1=0.8, n2=0.1)
    val_loader = DataLoader(val_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)
if run_testing:
    test_dataset = ChessDataset(db_path, n_limit, 'test', n1=0.8, n2=0.1)
    test_loader = DataLoader(test_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)









@dataclass
class Run_Config():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    batch_size: int = gpu_batch_size
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.01 * max_lr
    warmup_steps: float = 0.01
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




run_config = Run_Config()

max_lr = run_config.max_lr
min_lr = run_config.min_lr
warmup_steps = int(run_config.warmup_steps * run_config.total_steps) #see top
max_steps = int(run_config.max_steps * run_config.total_steps) #see top

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it/warmup_steps)
    elif it > warmup_steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


total_batch_size = run_config.total_batch_size # used for alphazero
batch_size = run_config.batch_size
assert total_batch_size % batch_size == 0
grad_accum_steps = total_batch_size // batch_size



optimizer = model.configure_optimizer(weight_decay=run_config.adamw_weight_decay, learning_rate=max_lr, device=device)
#torch.optim.AdamW(params, lr=max_lr, weight_decay=run_config.adamw_weight_decay, fused=use_fused) #no longer mode.parameters()


if save:
    model_path = os.path.join(model_dir, f"state_dict_v{iteration}.pth")

if write:
    log_path = os.path.join(model_dir, f"log_v{iteration}.txt")
    with open(log_path, 'w') as log_file:
        if existing_model_path is not None:
            log_file.write(f"Pretrained model, for {pretrained_data[0]} steps with bs {pretrained_data[1]} using ds {pretrained_data[2]} on iteration {pretrained_data[3]}\n")
        log_file.write(f"Iteration: {iteration}\n")

    with open(log_path, 'a') as log_file:
        log_file.write(f"Hyperparameters:\n")
        log_file.write(f"total_batch_size: {HyperParamConfig.total_batch_size}\n")
        log_file.write(f"adamw_weight_decay: {HyperParamConfig.adamw_weight_decay}\n")
        log_file.write(f"gradient_clipping: {HyperParamConfig.gradient_clipping}\n")
        log_file.write(f"max_lr: {HyperParamConfig.max_lr}\n")
        log_file.write(f"max_steps: {HyperParamConfig.max_steps}\n")
        log_file.write(f"n_layer: {HyperParamConfig.n_layer}\n")
        log_file.write(f"n_head: {HyperParamConfig.n_head}\n")
        log_file.write(f"n_embd: {HyperParamConfig.n_embd}\n")
        log_file.write(f"n_blocks_policyhead: {HyperParamConfig.n_blocks_policyhead}\n")
        log_file.write(f"n_blocks_valuehead: {HyperParamConfig.n_blocks_valuehead}\n")
        log_file.write(f"dropout: {HyperParamConfig.dropout}\n")
        log_file.write(f"total no of parameters: {total_params}\n")

def training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path):
    train_iter = iter(train_loader)
    loss_storage = {}
    print("starting training")
    for step in range(run_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        losses_list = []
        
        for micro_step in range(grad_accum_steps):
            try:
                board_state_tensor, special_token_tensor, target_p_tensor = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                board_state_tensor, special_token_tensor, target_p_tensor = next(train_iter)

            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)

            # Evaluate the loss
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x_policy, loss = model(board_state_tensor, special_token_tensor, target_p_tensor)
            if torch.isnan(loss):
                print(f"{loss=}")
            loss = loss / grad_accum_steps

            losses_list.append(loss.item())
            loss.backward()
            
        loss_accum = sum(losses_list)
        if math.isnan(loss_accum):
            print(grad_accum_steps)
            with open(debug_path, "a") as file:
                    for i in range(len(losses_list)):
                        file.write(f"{losses_list[i]}\n")
            import sys; sys.exit(0)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.gradient_clipping)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            if param_group['lr_type'] == -1: # policy head final linear layer
                param_group['lr'] = lr * 5e-2 # Smaller learning rate for final layers
            elif param_group['lr_type'] == -2: # policy head
                param_group['lr'] = lr * 5e-1  # Moderately smaller learning rate for entire policy and value heads
            else:
                param_group['lr'] = lr  # Default learning rate for the rest of the model

        optimizer.step()
        if log_path is not None:
            loss_storage[step] = loss_accum

        
        if step % 1000 == 0 or step == run_config.total_steps - 1:
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
                print(f"Model parameters saved to {model_path} at step {step}")
            if log_path is not None:
                with open(log_path, "a") as file:
                    for key, value in loss_storage.items():
                        file.write(f"step={key} | loss={value}\n")
                    file.write(f"Model parameters saved to {model_path} at step {step}\n")
                loss_storage = {}
        print(f"step={step} | loss={loss_accum}")
        # if step % 1000 == 999:
        #     validation(model, val_loader, device, run_config, log_path)
        
        
        


    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

def validation(model, val_loader, device, run_config, log_path):
    model.eval()
    losses_list = []
    val_iter = iter(val_loader)
    print("starting validation")
    with torch.no_grad():
        while True:
            try:
                board_state_tensor, special_token_tensor, target_p_tensor = next(val_iter)
            except StopIteration:
                break
            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)

            # Evaluate the loss
            x_policy, loss = model(board_state_tensor, special_token_tensor, target_p_tensor)
            losses_list.append(loss.item())
        loss_accum = sum(losses_list)/len(losses_list)
    if log_path is not None:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Validation Loss: {loss_accum}\n")

    print(f"Validation Loss: {loss_accum}")

if run_training:
    training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path)

if run_validation:
    validation(model, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path)


# if __name__ == '__main__':
#     main()
