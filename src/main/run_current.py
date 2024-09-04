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
db_path = '/workspace/model/databases/TCEC_github_v2.db'
model_dir = "/workspace/runs/full_TCEC_run_1/iters"
best_model_path = "/workspace/runs/full_TCEC_run_1/best_model.pth"
model_path = None #correct path set ltr if save == True
log_path = None #correct path set ltr if write == True
debug_path = "debug.txt"

iteration = 0
existing_model_path = None #"../runs/full_TCEC_run_1/iters/state_dict_v41.pth"
pretrained_data = (43000, 4096, "TCEC_github_v2.db", 41) #(steps_completed, batch_size, db, iteration no of model we are loading)

run_training = True
run_validation = True
run_testing = False
write = True
save = True
actual_lr = 1e-3

with open("/workspace/runs/full_TCEC_run_1/completed_indices.txt", "a+") as file:
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


train_steps = 30000
n_workers = 16
n1 = 0.8
n2 = 0.1
gpu_batch_size = 4096

#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int = 4096
    adamw_weight_decay: float = 1e-3
    gradient_clipping: float = 1.0
    max_lr: float = 1e-3
    max_steps: float = 0.80
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 128
    n_blocks_policyhead: int = 3
    n_blocks_valuehead: int = 4
    dropout: float = 0.3

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
        #x = self.output_tanh(x)
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
            state_dict = torch.load(existing_model_path)
            adjusted_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()} #torch.compile is adding the prefix "_orig_mod." to all keys of the state_dict for some reason, need to remove it
            self.load_state_dict(adjusted_state_dict)
        else:
            print("initialising new model")
            self.apply(self._init_weights)

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


    def forward(self, data, legal_indices, p=None, v=None):
        # idx is of shape (B, T)
        B, T = data.size()
        x = self.transformer(data)

        x_policy = self.policy_head(x, legal_indices)
        x_value = self.value_head(x)
        x_value = x_value.squeeze()
        loss = None
        if p is not None and v is not None:
            p = p.long()
            x_policy_softmax = F.softmax(x_policy, dim=1)
            correct_index_values = torch.full((x_policy_softmax.shape[0],), -1.0, dtype=torch.double).to(device)

            for b in range(x_policy_softmax.shape[0]):
                correct_index_values[b] = x_policy_softmax[b][p[b]]
            

            loss_p = F.cross_entropy(x_policy, p)
            loss_v = F.mse_loss(x_value, v)
            #loss_v = F.l1_loss(x_value,v)
            #loss = loss_v + loss_p
        return x_policy, x_value, loss_p, loss_v

class ChessDataset(Dataset):
    def __init__(self, db_path, split, n1=0.8, n2=0.1, transform=None):
        self.transform = transform
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
        x = self.data[0][idx]
        p = self.data[1][idx]
        v = self.data[2][idx]
        y = self.data[3][idx]

        if self.transform:
            x = self.transform(x)

        return x, p, v, y

    def get_dataframe(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT move_index, legal_move_indices, x, evaluation FROM evaluations" # WHERE n_pieces <= 12" #Change number of pieces per position
        cursor.execute(query)
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=['move_index', 'legal_move_indices', 'x', 'evaluation'])
        df = df.drop_duplicates(subset=['move_index', 'legal_move_indices', 'x', 'evaluation'], keep='first')
        print("fetched df")
        return df

    def convert_df(self, start_index, end_index):
        x, p, v, y = [], [], [], []
        max_length_y = 0

        ave_legal_index_len = 0

        #for _, row in self.df.iterrows():
        for index, row in self.df.iloc[start_index:end_index].iterrows():
            legal_indices = row['legal_move_indices'].split(',')[:-1]
            legal_indices = [int(idx) for idx in legal_indices]
            ave_legal_index_len += len(legal_indices)
            if len(legal_indices) > max_length_y:
                max_length_y = len(legal_indices)
        ave_legal_index_len /= (end_index - start_index)
        print(f"{ave_legal_index_len=}")
        print("found y-padding value")
        for index, row in self.df.iloc[start_index:end_index].iterrows():
            x_add = row['x'].split(',')[:-1]
            x_add = [int(xi) for xi in x_add]

            move_index = row['move_index']
            evaluation = row['evaluation']
            legal_indices = row['legal_move_indices'].split(',')[:-1]
            legal_indices = [int(idx) for idx in legal_indices]

            if len(legal_indices) < max_length_y:
                legal_indices.extend([-1] * (max_length_y - len(legal_indices)))

            x.append(x_add)
            p.append(move_index)
            v.append(evaluation)
            y.append(legal_indices)

            if index % 1000000 == 0:
                print(f"processed {index} rows")

        print("finished iterating through all rows")

        tanh_layer = nn.Tanh()
        #sigmoid_layer = nn.Sigmoid()

        x = torch.tensor(x)
        p = torch.tensor(p)
        # for i, v_value in enumerate(v):
        #     if v_value > 10:
        #         v[i] = 5
        #     elif v_value < -10:
        #         v[i] = -5
        #     else:
        #         v[i] *= 0.5
        v = torch.tensor(v) * 0.2
        v = tanh_layer(v) 
        y = torch.tensor(y)

        return x, p, v, y





#indv batch size always 16 (or as much as GPU can handle)
if run_training:
    train_dataset = ChessDataset(db_path, split='train', n1=0.8, n2=0.1)
    train_loader = DataLoader(train_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)
if run_validation:
    val_dataset = ChessDataset(db_path, split='val', n1=0.8, n2=0.1)
    val_loader = DataLoader(val_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)
if run_testing:
    test_dataset = ChessDataset(db_path, split='test', n1=0.8, n2=0.1)
    test_loader = DataLoader(test_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)









@dataclass
class Run_Config():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    batch_size: int = gpu_batch_size
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.1 * max_lr
    warmup_steps: float = 0.333
    max_steps: float = HyperParamConfig.max_steps #[0.70, 0.75, 0.80]
    total_steps: int = train_steps #2 epochs for bay optim

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

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chess(Chess_Config())
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
model = torch.compile(model)
policy_params = sum(p.numel() for name, p in model.named_parameters() if 'policy' in name and p.requires_grad)
print(f"Total number of parameters in the policy head: {policy_params}")
# Assuming 'model.value_head' is the submodule for your value head
value_params = sum(p.numel() for p in model.value_head.parameters() if p.requires_grad)
print(f"Total number of parameters in the value head: {value_params}")




run_config = Run_Config()

max_lr = run_config.max_lr
min_lr = run_config.min_lr
warmup_steps = int(run_config.warmup_steps * run_config.total_steps) #see top
max_steps = int(run_config.max_steps * run_config.total_steps) #see top

def get_lr(it):
    return actual_lr
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


policy_head_fc_params = list(model.policy_head.fc.parameters())
policy_head_fc_param_ids = {id(param) for param in policy_head_fc_params}
value_head_fc_params = list(model.value_head.fc.parameters())
value_head_fc_params_ids = {id(param) for param in value_head_fc_params}

policy_head_params = [param for name, param in model.named_parameters() if 'policy_head' in name and id(param) not in policy_head_fc_param_ids]
value_head_params = [param for name, param in model.named_parameters() if 'value_head' in name and id(param) not in value_head_fc_params_ids]

rest_of_model_params = [param for name, param in model.named_parameters() if 'policy_head' not in name and 'value_head' not in name]

params = [
    {'params': policy_head_fc_params, 'lr_type': -1},  # Final layer of policy head
    {'params': value_head_fc_params, 'lr_type': -3},   # Final layer of value head
    {'params': policy_head_params, 'lr_type': -2},  # Entire policy head except final layer
    {'params': value_head_params, 'lr_type': -4},  # Entire value head except final layer
    {'params': rest_of_model_params, 'lr_type': 1}  # Rest of the model
]

optimizer = torch.optim.AdamW(params, lr=max_lr, weight_decay=run_config.adamw_weight_decay) #no longer mode.parameters()


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
    consecutive_counter = 0

    loss_storage = {}
    print("starting training")
    for step in range(run_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        losses_list = []
        losses_p_list = []
        losses_v_list = []
        updated_consecutive_counter = False
        
        for micro_step in range(grad_accum_steps):
            try:
                data, p, v, legal_indices = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data, p, v, legal_indices = next(train_iter)

            data, p, v, legal_indices = data.to(device), p.to(device), v.to(device), legal_indices.to(device)

            # Evaluate the loss
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x_policy, x_value, loss_p, loss_v = model(data, legal_indices, p, v)
            loss = loss_p + loss_v
            if torch.isnan(loss):
                print(f"{loss=}")
            loss = loss / grad_accum_steps
            #print(loss)
            #import sys; sys.exit(0)
            
            losses_list.append(loss.item())
            losses_p_list.append(loss_p.item())
            losses_v_list.append(loss_v.item())
            loss.backward()
            
        loss_accum = sum(losses_list)
        loss_p_accum = sum(losses_p_list)
        loss_v_accum = sum(losses_v_list)
        if math.isnan(loss_accum):
            print(grad_accum_steps)
            with open(debug_path, "a") as file:
                    for i in range(len(losses_list)):
                        file.write(f"{losses_list[i]}\n")
            import sys; sys.exit(0)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.gradient_clipping)
        lr = get_lr(step+15000)
        for param_group in optimizer.param_groups:
            if param_group['lr_type'] == -1: # policy head final linear layer
                param_group['lr'] = lr * 5e-2 # Smaller learning rate for final layers
            elif param_group['lr_type'] == -2: # policy head
                param_group['lr'] = lr * 5e-1  # Moderately smaller learning rate for entire policy and value heads
            elif param_group['lr_type'] == -3: # value head final linear layer
                param_group['lr'] = lr * 5e-4
            elif param_group['lr_type'] == -4: # value head
                param_group['lr'] = lr * 5e-3
            else:
                param_group['lr'] = lr  # Default learning rate for the rest of the model

        optimizer.step()
        if log_path is not None:
            loss_storage[step] = (loss_accum, loss_p_accum, loss_v_accum)

        
        if step % 100 == 0 or step == run_config.total_steps - 1:
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
                print(f"Model parameters saved to {model_path} at step {step}")
            if log_path is not None:
                with open(log_path, "a") as file:
                    for key, value in loss_storage.items():
                        file.write(f"step={key} | loss={value[0]} | loss_p={value[1]} | loss_v={value[2]}\n")
                    file.write(f"Model parameters saved to {model_path} at step {step}\n")
                loss_storage = {}
        if step % 1000 == 999:
            validation(model, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path)
        print(f"step={step} | loss={loss_accum} | loss_p={loss_p_accum} | loss_v={loss_v_accum}")
        
        


    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

def validation(model, train_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path):
    model.eval()
    val_loss = 0
    losses_list = []
    losses_p_list = []
    losses_v_list = []
    val_iter = iter(val_loader)
    print("starting validation")
    with torch.no_grad():
        while True:
            try:
                data, p, v, legal_indices = next(val_iter)
            except StopIteration:
                break
            data, p, v, legal_indices = data.to(device), p.to(device), v.to(device), legal_indices.to(device)

            # Evaluate the loss
            x_policy, x_value, loss_p, loss_v = model(data, legal_indices, p, v)
            loss = loss_p + loss_v
            losses_list.append(loss.item())
            losses_p_list.append(loss_p.item())
            losses_v_list.append(loss_v.item())
        loss_accum = sum(losses_list)/len(losses_list)
        loss_p_accum = sum(losses_p_list)/len(losses_p_list)
        loss_v_accum = sum(losses_v_list)/len(losses_v_list)
    if log_path is not None:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Validation Loss: {loss_accum} | loss_p={loss_p_accum} | loss_v={loss_v_accum}\n")

    print(f"Validation Loss: {loss_accum} | loss_p={loss_p_accum} | loss_v={loss_v_accum}")

if run_training:
    training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path)

if run_validation:
    validation(model, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path)


# if __name__ == '__main__':
#     main()


