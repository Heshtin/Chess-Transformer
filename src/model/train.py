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

from chess_model import Chess
from dataloader import ChessIterableDataset, pad_collate
from dataclass import Run_Config, Chess_Config

db_path = '/workspace/database/lichess_2/combined_database.db'
model_dir = "/workspace/runs/lichess_run/iters"
best_model_path = "/workspace/runs/lichess_run/best_model.pth"
model_path = None #correct path set ltr if save == True
log_path = None #correct path set ltr if write == True
debug_path = "debug.txt"

iteration = 0
existing_model_path = "../../runs/lichess_run/iters/state_dict_v53.pth"
pretrained_data = (75000, 1024, "combined_database.db", 53) #(steps_completed, batch_size, db, iteration no of model we are loading)

run_training = True
run_validation = False
run_testing = False
write = True
save = True

constant_lr = 4e-4
global_train_type = "normal" # normal or reevaluation
global_masking = False
# if global_masking:
#     torch._dynamo.config.suppress_errors = True
train_steps = 150000
global_n_limit = None
n_workers = 1
global_n1 = 0.8
global_n2 = 0.1
gpu_batch_size = 1024



#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int = 1024
    adamw_weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    max_lr: float = 5e-3
    constant_lr: 
    max_steps: float = 0.80
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0


with open("/workspace/runs/lichess_run/completed_indices.txt", "a+") as file:
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

def training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path, train_type, masking):
    train_iter = iter(train_loader)
    loss_storage = {}
    print("starting training")
    forward_pass = "first"
    reevaluation_moves_tensor = None
    for step in range(run_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        losses_list = []
        losses_p_list = []
        losses_rp_list = []
        
        for micro_step in range(grad_accum_steps):
            
            try:
                board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor = next(train_iter)
            if masking:
                legal_moves_tensor = legal_moves_tensor.to(device)
            #     print(f"{legal_moves_tensor=}")
            # print(f"{board_state_tensor=}")
            # print(f"{special_token_tensor=}")
            # print(f"{target_p_tensor=}")
            # #import sys; sys.exit(0)
            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)

            # Evaluate the loss
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x_policy, loss_p, loss_rp = model(board_state_tensor, special_token_tensor, target_p_tensor, train_type, legal_moves_tensor)
                loss = loss_p + loss_rp if loss_rp is not None else loss_p.clone() # if training is normal (not reevaluate) then loss_rp=None
            # if torch.isnan(loss):
            #     print(f"{loss=}")
            
            loss = loss / grad_accum_steps
            losses_list.append(loss.item())
            

            if loss_rp:
                loss_p = loss_p / grad_accum_steps
                losses_p_list.append(loss_p.item())
                loss_rp = loss_rp / grad_accum_steps
                losses_rp_list.append(loss_rp.item())
            loss.backward()
            
        loss_accum = sum(losses_list)
        if loss_rp:
            loss_accum_p = sum(losses_p_list)
            loss_accum_rp = sum(losses_rp_list)
        if math.isnan(loss_accum):
            print(grad_accum_steps)
            with open(debug_path, "a") as file:
                    for i in range(len(losses_list)):
                        file.write(f"{losses_list[i]}\n")
            import sys; sys.exit(0)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.gradient_clipping)
        lr = constant_lr if constant_lr else get_lr(step)
        for param_group in optimizer.param_groups:
            if param_group['lr_type'] == -1: # policy head final linear layer
                param_group['lr'] = lr * 0.1 #* 5e-2 # Smaller learning rate for final layers
            elif param_group['lr_type'] == -2: # policy head
                param_group['lr'] = lr #* 5e-1  # Moderately smaller learning rate for entire policy and value heads
            else:
                param_group['lr'] = lr  # Default learning rate for the rest of the model

        optimizer.step()
        if log_path is not None:
            if loss_rp:
                loss_storage[step] = (loss_accum, loss_accum_p, loss_accum_rp)
            else:
                loss_storage[step] = loss_accum

        
        if step % 1000 == 0 or step == run_config.total_steps - 1:
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
                print(f"Model parameters saved to {model_path} at step {step}")
            if log_path is not None:
                with open(log_path, "a") as file:
                    for key, value in loss_storage.items():
                        if loss_rp:
                            file.write(f"step={key} | loss={value[0]} | loss_p={value[1]} | loss_rp={value[2]}\n")
                        else:
                            file.write(f"step={key} | loss={value}")
                    file.write(f"Model parameters saved to {model_path} at step {step}\n")
                loss_storage = {}
        if loss_rp:
            print(f"step={step} | loss={loss_accum} | loss_p={loss_accum_p} | loss_rp={loss_accum_rp}")
        else:
            print(f"step={step} | loss={loss_accum}")
        if step % 10000 == 9999 and run_validation:
            validation(model, val_loader, device, run_config, log_path)
        
        
        


    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

def validation(model, val_loader, device, run_config, log_path):
    model.eval()
    losses_list = []
    losses_p_list = []
    losses_rp_list = []
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
            x_policy, loss_p, loss_rp = model(board_state_tensor, special_token_tensor, target_p_tensor)
            loss = loss_p + loss_rp
            losses_list.append(loss.item())
            losses_p_list.append(loss_p.item())
            losses_rp_list.append(loss_rp.item())
        loss_accum = sum(losses_list)/len(losses_list)
        loss_accum_p = sum(losses_p_list)/len(losses_p_list)
        loss_accum_rp = sum(losses_rp_list)/len(losses_rp_list)
    if log_path is not None:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Validation Loss: loss={loss_accum} | loss_p={loss_accum_p} | loss_rp={loss_accum_rp}\n")

    print(f"Validation Loss: loss={loss_accum} | loss_p={loss_accum_p} | loss_rp={loss_accum_rp}")

if run_training:
    training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, run_config, model_path, log_path, global_train_type, global_masking)

if run_validation:
    validation(model, val_loader, device, run_config, log_path)


# if __name__ == '__main__':
#     main()
