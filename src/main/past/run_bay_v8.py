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
#import matplotlib.pyplot as plt

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
model_dir = "/workspace/runs/bay_optim_run_1/iters"
checkpoint_path = "/workspace/runs/bay_optim_run_1/optimizer_v53.pkl" #for bayesian optim model
best_model_path = "/workspace/runs/bay_optim_run_1/best_model_2.pth"

#model_path = os.path.join(model_dir, "test.pth")

#tensor_dict_path = "tensor.pth"
# bash_command = 'bash /workspace/model/bash_files/transfer_and_shutdown.sh'

#bay optim
n_calls = 50
checkpoint_interval = 1  # Save checkpoint every 5 iterations
increment_size = 500000     # Increase dataset size by 200 every 5 iterations
initial_dataset_size = 1000000  # Start with a smaller subset

train_steps = 10
n_workers = 12
n1 = 0.8
n2 = 0.1
gpu_batch_size = 128

param_space = [
    Categorical([4096], name='total_batch_size'),  # Specific values
    Categorical([1e-5], name='adamw_weight_decay'),
    #Real(1e-5, 1e-4, prior='log-uniform', name='adamw_weight_decay'),  # Continuous range with log-uniform distribution
    Real(0.99, 1.0, name='gradient_clipping'),  # Continuous range
    Categorical([1e-8], name='max_lr'),
    #Real(1e-3, 1e-2, prior='log-uniform', name='max_lr'),  # Continuous range with log-uniform distribution
    Real(0.89, 0.90, name='max_steps'),  # Specific values
    Categorical([16], name='n_layer'),  # Specific values
    Categorical([8], name='n_head'),  # Specific values
    Categorical([128], name='n_embd'),  # Specific values
    Categorical([3], name='n_blocks_policyhead'),  # Specific values
    Categorical([4], name='n_blocks_valuehead'),  # Specific values
    Real(0.2, 0.3, name='dropout')  # Continuous range
]

# param_space = [
#     Categorical([4096], name='total_batch_size'),  # Specific values
#     Real(1e-8, 1e-7, prior='log-uniform', name='adamw_weight_decay'),  # Continuous range with log-uniform distribution
#     Real(0.1, 1.0, name='gradient_clipping'),  # Continuous range
#     Real(1e-4, 1e-3, prior='log-uniform', name='max_lr'),  # Continuous range with log-uniform distribution
#     Real(0.80, 0.90, name='max_steps'),  # Specific values
#     Categorical([8, 16, 32], name='n_layer'),  # Specific values
#     Categorical([8, 16, 32], name='n_head'),  # Specific values
#     Categorical([128, 256, 512], name='n_embd'),  # Specific values
#     Categorical([2, 3, 4], name='n_blocks_policyhead'),  # Specific values
#     Categorical([2, 3, 4], name='n_blocks_valuehead'),  # Specific values
#     Real(0.1, 0.2, name='dropout')  # Continuous range
# ]


#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int
    adamw_weight_decay: float
    gradient_clipping: float
    max_lr: float
    max_steps: float
    n_layer: int
    n_head: int
    n_embd: int
    n_blocks_policyhead: int
    n_blocks_valuehead: int
    dropout: float

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
            mask[b, batch_tensor] = 0
        masked_input = x + mask
        output = masked_input
        #output = F.softmax(masked_input, dim=1)
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
            
            # with open("policy_vector.txt", "a") as file:
            #     for b in x_policy:
            #         file.write(str(b.tolist()))
            #         file.write(",\n")
            #     file.write(str(p))
            #     print("written")
            #import sys; sys.exit(0)
            loss_p = F.cross_entropy(x_policy, p)
            #loss_p = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_v = F.mse_loss(x_value, v)
            loss = loss_p
        return x_policy, x_value, loss

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
        query = "SELECT move_index, legal_move_indices, x, evaluation FROM evaluations WHERE n_pieces <= 12" #Change number of pieces per position
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

        m = nn.Tanh()

        x = torch.tensor(x)
        p = torch.tensor(p)
        v = torch.tensor(v)
        v = m(v)
        y = torch.tensor(y)

        return x, p, v, y

train_dataset = ChessDataset(db_path, split='train', n1=0.8, n2=0.1)
val_dataset = ChessDataset(db_path, split='val', n1=0.8, n2=0.1)
test_dataset = ChessDataset(db_path, split='test', n1=0.8, n2=0.1)

#indv batch size always 16 (or as much as GPU can handle)
train_loader = DataLoader(train_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)
val_loader = DataLoader(val_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)
#test_loader = DataLoader(test_dataset, batch_size=gpu_batch_size, shuffle=False, num_workers=n_workers)








def train_and_evaluate(HyperParamConfig, DataSize, iteration):
    @dataclass
    class Run_Config():
        total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
        batch_size: int = gpu_batch_size
        adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
        gradient_clipping = HyperParamConfig.gradient_clipping
        max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
        min_lr: float = 0.1 * max_lr
        warmup_steps: float = 0.1
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
    model = torch.compile(model)


    run_config = Run_Config()

    max_lr = run_config.max_lr
    min_lr = run_config.min_lr
    warmup_steps = int(run_config.warmup_steps * run_config.total_steps) #see top
    max_steps = int(run_config.max_steps * run_config.total_steps) #see top

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (max_steps - warmup_steps)
        elif it > warmup_steps:
            return min_lr
        else:
            decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)

    def excl_outlier_loss(losses):
        Q1 = np.percentile(losses, 25)
        Q3 = np.percentile(losses, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_losses = [loss for loss in losses if lower_bound <= loss <= upper_bound]

        return filtered_losses


    total_batch_size = run_config.total_batch_size # used for alphazero
    batch_size = run_config.batch_size
    assert total_batch_size % batch_size == 0
    grad_accum_steps = total_batch_size // batch_size
    
    # for name, param in model.named_parameters():
    #     print(name)


    # params = [
    # {'params': [param for name, param in model.named_parameters() if 'policy_head' in name and 'fc' not in name], 'lr_type': -1},  # Final layer of policy head
    # {'params': [param for name, param in model.named_parameters() if 'value_head' in name and 'fc' not in name], 'lr_type': -1},   # Final layer of value head
    # {'params': [param for name, param in model.named_parameters() if 'policy_head.fc' in name or 'value_head.fc' in name], 'lr_type': 0},  # Entire policy and value heads except final layers
    # {'params': [param for name, param in model.named_parameters() if 'policy_head.fc' not in name and 'value_head.fc' not in name], 'lr_type': 1}  # Rest of the model
    # ]

    policy_head_fc_params = list(model.policy_head.fc.parameters())
    policy_head_fc_param_ids = {id(param) for param in policy_head_fc_params}
    value_head_fc_params = list(model.value_head.fc.parameters())
    value_head_fc_params_ids = {id(param) for param in value_head_fc_params}

    policy_head_params = [param for name, param in model.named_parameters() if 'policy_head' in name and id(param) not in policy_head_fc_param_ids]
    value_head_params = [param for name, param in model.named_parameters() if 'value_head' in name and id(param) not in value_head_fc_params_ids]

    rest_of_model_params = [param for name, param in model.named_parameters() if 'policy_head' not in name and 'value_head' not in name]

    params = [
        {'params': policy_head_fc_params, 'lr_type': -1},  # Final layer of policy head
        {'params': value_head_fc_params, 'lr_type': -1},   # Final layer of value head
        {'params': policy_head_params, 'lr_type': -2},  # Entire policy head except final layer
        {'params': value_head_params, 'lr_type': -2},  # Entire value head except final layer
        {'params': rest_of_model_params, 'lr_type': 1}  # Rest of the model
    ]

    optimizer = torch.optim.AdamW(params, lr=max_lr, weight_decay=run_config.adamw_weight_decay) #no longer mode.parameters()
    print("loading data")

    model_path = os.path.join(model_dir, f"state_dict_v{iteration}.pth")
    log_path = os.path.join(model_dir, f"log_v{iteration}.txt")

    with open(log_path, 'w') as log_file:
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

    #for step in range(run_config.total_steps):

    train_iter = iter(train_loader)
    consecutive_counter = 0

    
    loss_storage = {}

    for step in range(run_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        losses_list = []
        updated_consecutive_counter = False
        
        for micro_step in range(grad_accum_steps):
            #data, p, v, legal_indices = data_loader.get_batch('train')
            try:
                data, p, v, legal_indices = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data, p, v, legal_indices = next(train_iter)

            data, p, v, legal_indices = data.to(device), p.to(device), v.to(device), legal_indices.to(device)

            # Evaluate the loss
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x_policy, x_value, loss = model(data, legal_indices, p, v)

            

            # x_policy = torch.sum(x_policy,dim= 0)
            # if micro_step == 0:
            #     avr_policy = torch.zeros_like(x_policy)
            # avr_policy +=x_policy
            
        

            loss = loss / grad_accum_steps
            losses_list.append(loss.item())
            loss.backward()
        loss_accum = sum(losses_list)
        #avr_policy = avr_policy/grad_accum_steps
        # with open("policy_vector.txt","a") as file:
        #     file.write(str(avr_policy.tolist()))
        #sys.exit(0)

        # excl_outliers_losses = excl_outlier_loss(losses_list)
        # loss_excl_outliers = sum(excl_outliers_losses) * (grad_accum_steps / len(excl_outliers_losses))

        # if (step >= 5 and loss_excl_outliers > 1000.0) or (step >= 20 and loss_excl_outliers > 100.0) or (step >= 100 and loss_excl_outliers > 20.0) or (step >= 1000 and loss_excl_outliers > 10.0):
        #     if consecutive_counter >= 5:
        #         break
        #     else:
        #         consecutive_counter += 1
        #         updated_consecutive_counter = True

        # if not updated_consecutive_counter and consecutive_counter > 0:
        #     consecutive_counter = 0

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.gradient_clipping)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            if param_group['lr_type'] == -1:
                param_group['lr'] = lr * 1e-2 # Smaller learning rate for final layers
            elif param_group['lr_type'] == -2:
                param_group['lr'] = lr * 1e-1  # Moderately smaller learning rate for entire policy and value heads
            else:
                param_group['lr'] = lr  # Default learning rate for the rest of the model



        # lr = get_lr(step)
        # for param_group in optimizer.param_groups:
        #     if param_group['initial_lr'] == -1:
        #         param_group['lr'] = lr * 1e-10  # Smaller learning rate for final layers
        #     else:
        #         param_group['lr'] = lr

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        #if not math.isnan(loss_accum):
        optimizer.step()
        loss_storage[step] = loss_accum

        
        if step % 100 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model parameters saved to {model_path} at step {step}")
            with open(log_path, "a") as file:
                for key, value in loss_storage.items():
                    file.write(f"step={key} | loss={value}\n")
                    file.write(f"Model parameters saved to {model_path} at step {step}")
            loss_storage = {}
        print(f"step={step} | loss={loss_accum}")
        



    #model_path = "/home/ubuntu/TCEC_github_params_v1.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")


    model.eval()
    val_loss = 0
    losses_list = []
    val_iter = iter(val_loader)
    with torch.no_grad():
        while True:
            try:
                data, p, v, legal_indices = next(val_iter)
                # Move data to the correct device
                data, p, v, legal_indices = data.to(device), p.to(device), v.to(device), legal_indices.to(device)
                # Forward pass through the model
                x_policy, x_value, loss = model(data, legal_indices, p, v)
                
                # Collect the loss
                losses_list.append(loss.item())
            except StopIteration:
                break
        #for data, p, v, legal_indices in val_loader:
            #data, p, v, legal_indices = data.to(device), p.to(device), v.to(device), legal_indices.to(device)

            # Evaluate the loss
            
        #val loss
        # excl_outliers_losses = excl_outlier_loss(losses_list)
        # loss_excl_outliers = sum(excl_outliers_losses) / len(excl_outliers_losses)
        loss_accum = np.mean(losses_list) if losses_list else 0

    with open(log_path, 'a') as log_file:
        log_file.write(f"Validation Loss: {loss_accum}\n")

    print(f"Validation Loss: {loss_accum}")
    return loss_accum, model


    # #os.system(bash_command)
    # print(f"copied parameters file to local_host")

def save_checkpoint(optimizer, path):
    try:
        print(f"Saving checkpoint to {path}...")
        joblib.dump(optimizer, path)
        print(f"Checkpoint successfully saved to {path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

# Function to load the optimizer state
def load_checkpoint(path):
    try:
        if not os.path.exists(path):
            print(f"Checkpoint file {path} does not exist.")
            return None
        print(f"Loading checkpoint from {path}...")
        optimizer = joblib.load(path)
        print(f"Checkpoint successfully loaded from {path}")
        return optimizer
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None




# Define the objective function for gp_minimize
def make_objective(initial_dataset_size, checkpoint_interval, increment_size):
    def objective(params, iteration):
        hyperparam_config = HyperParamConfig(
            total_batch_size=params[0],
            adamw_weight_decay=params[1],
            gradient_clipping=params[2],
            max_lr=params[3],
            max_steps=params[4],
            n_layer=params[5],
            n_head=params[6],
            n_embd=params[7],
            n_blocks_policyhead=params[8],
            n_blocks_valuehead=params[9],
            dropout=params[10]
        )


        dataset_size = initial_dataset_size + (iteration // checkpoint_interval) * increment_size
        return train_and_evaluate(hyperparam_config, dataset_size, iteration)
    return objective



def main():
    #global best_loss

    # Load checkpoint if available
    optimizer = load_checkpoint(checkpoint_path)
    if optimizer:
        start_iter = len(optimizer.Xi)
        #best_loss = min(optimizer.yi) if optimizer.yi else float('inf')
    else:
        optimizer = Optimizer(
            dimensions=param_space,
            random_state=42
        )
        start_iter = 0

    objective_fn = make_objective(initial_dataset_size, checkpoint_interval, increment_size)

    for i in range(start_iter, n_calls):
        params = optimizer.ask()

        # Call the objective function
        loss, model = objective_fn(params, i)

        # Update the optimizer with the new observation
        optimizer.tell(params, loss)

        # Save checkpoint at regular intervals
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(optimizer, checkpoint_path)
            print(f"Checkpoint saved at iteration {i + 1}")

    # Save final checkpoint
    save_checkpoint(optimizer, checkpoint_path)
    print("Final checkpoint saved")

    # Get the best found parameters
    #best_params = result.x
    best_params = optimizer.Xi[optimizer.yi.index(min(optimizer.yi))]
    print("Best hyperparameters found: ", best_params)

    best_hyperparam_config = HyperParamConfig(
        total_batch_size=best_params[0],
        adamw_weight_decay=best_params[1],
        gradient_clipping=best_params[2],
        max_lr=best_params[3],
        max_steps=best_params[4],
        n_layer=best_params[5],
        n_head=best_params[6],
        n_embd=best_params[7],
        n_blocks_policyhead=best_params[8],
        n_blocks_valuehead=best_params[9],
        dropout=best_params[10]
    )

    _, best_model = train_and_evaluate(best_hyperparam_config, initial_dataset_size, start_iter)
    torch.save(best_model.state_dict(), best_model_path)
    print(f"Best model saved with hyperparameters: {best_hyperparam_config}")

if __name__ == '__main__':
    main()


