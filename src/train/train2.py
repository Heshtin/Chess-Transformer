import sys
import os
import importlib
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset

torch.manual_seed(1337)  #pytorch seed
np.random.seed(1337) #numpy seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) #main GPU seed 
    torch.cuda.manual_seed_all(1337) #multi-GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataloader import ChessIterableDataset, pad_collate
from dataclass import VariableRunConfig, DataConfig, HyperParamConfig, RunConfig, ChessConfig
from auxilliary import retrieve_iteration_number, write_to_hyperparam

sys.path.append('../models/model0')
import chess_model
importlib.reload(chess_model)  # Reloads the module
from chess_model import Chess  # Now import the class

db_path = '/workspace/database/lichess_2/combined_database.db'
model_dir = "/workspace/runs/lichess_run/iters"
best_model_path = "/workspace/runs/lichess_run/best_model.pth"
model_path = None #correct path set ltr if save == True
log_path = None #correct path set ltr if write == True
debug_path = "debug.txt"
existing_model_path = None #"../../runs/lichess_run/iters/state_dict_v53.pth"

pretrained_data = (75000, 1024, "combined_database.db", 53) #(steps_completed, batch_size, db, iteration no of model we are loading)

run_training = True
run_validation = False
run_testing = False
write = False
save = False

iteration = retrieve_iteration_number(write)
print(f"{iteration=}")

train_loader = None
val_loader = None
test_loader = None

if run_training:
    train_dataset = ChessIterableDataset(db_path, 'train', VariableRunConfig.n_limit, DataConfig.n1, DataConfig.n2, VariableRunConfig.masking)
    train_loader = DataLoader(train_dataset, batch_size=VariableRunConfig.gpu_batch_size, num_workers=DataConfig.n_workers, collate_fn=pad_collate)
if run_validation:
    val_dataset = ChessIterableDataset(db_path, 'val', VariableRunConfig.n_limit, DataConfig.n1, DataConfig.n2, VariableRunConfig.masking)
    val_loader = DataLoader(val_dataset, batch_size=VariableRunConfig.gpu_batch_size, num_workers=DataConfig.n_workers, collate_fn=pad_collate)
if run_testing:
    test_dataset = ChessIterableDataset(db_path, 'test', VariableRunConfig.n_limit, DataConfig.n1, DataConfig.n2, VariableRunConfig.masking)
    test_loader = DataLoader(test_dataset, batch_size=VariableRunConfig.gpu_batch_size, num_workers=DataConfig.n_workers, collate_fn=pad_collate)

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Chess(ChessConfig())
model.to(device)
print(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ", total_params)
# If your policy head parameters are scattered in the model
policy_params = sum(p.numel() for name, p in model.named_parameters() if 'policy' in name and p.requires_grad)
print(f"Total number of parameters in the policy head: {policy_params}")

model = torch.compile(model)


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it/warmup_steps)
    elif it > warmup_steps:
        return min_lr
    else:
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

max_lr = RunConfig.max_lr
min_lr = RunConfig.min_lr
warmup_steps = int(RunConfig.warmup_steps * RunConfig.total_steps) #see top
max_steps = int(RunConfig.max_steps * RunConfig.total_steps) #see top

total_batch_size = RunConfig.total_batch_size # used for alphazero
batch_size = VariableRunConfig.gpu_batch_size
assert total_batch_size % batch_size == 0
grad_accum_steps = total_batch_size // batch_size

optimizer = model.configure_optimizer(weight_decay=RunConfig.adamw_weight_decay, learning_rate=max_lr, device=device)
#torch.optim.AdamW(params, lr=max_lr, weight_decay=RunConfig.adamw_weight_decay, fused=use_fused) #no longer mode.parameters()


if save:
    model_path = os.path.join(model_dir, f"state_dict_v{iteration}.pth")

if write:
    log_path = os.path.join(model_dir, f"log_v{iteration}.txt")
    with open(log_path, 'w') as log_file:
        if existing_model_path is not None:
            log_file.write(f"Pretrained model, for {pretrained_data[0]} steps with bs {pretrained_data[1]} using ds {pretrained_data[2]} on iteration {pretrained_data[3]}\n")
        log_file.write(f"Iteration: {iteration}\n")

    write_to_hyperparam(log_path, total_params, HyperParamConfig())


def training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, RunConfig, model_path, log_path, train_type, masking):
    train_iter = iter(train_loader)
    loss_storage = {}
    print("starting training")
    for step in range(RunConfig.total_steps):
        optimizer.zero_grad(set_to_none=True)
        losses_list = []        
        for micro_step in range(grad_accum_steps):
            try:
                board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                board_state_tensor, special_token_tensor, target_p_tensor, legal_moves_tensor = next(train_iter)
            if masking:
                legal_moves_tensor = legal_moves_tensor.to(device)
            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                x_policy = model(board_state_tensor, special_token_tensor, target_p_tensor, train_type, legal_moves_tensor)
                loss = F.cross_entropy(x_policy, target_p_tensor)
            
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
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RunConfig.gradient_clipping)
        lr = HyperParamConfig.constant_lr if HyperParamConfig.constant_lr else get_lr(step)
        for param_group in optimizer.param_groups:
            if param_group['lr_type'] == -1: # policy head final linear layer
                param_group['lr'] = lr * 0.1 #* 5e-2 # Smaller learning rate for final layers
            elif param_group['lr_type'] == -2: # policy head
                param_group['lr'] = lr #* 5e-1  # Moderately smaller learning rate for entire policy and value heads
            else:
                param_group['lr'] = lr  # Default learning rate for the rest of the model

        optimizer.step()
        if log_path is not None:
            loss_storage[step] = loss_accum
        
        if step % 1000 == 0 or step == RunConfig.total_steps - 1:
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
                print(f"Model parameters saved to {model_path} at step {step}")
            if log_path is not None:
                with open(log_path, "a") as file:
                    for key, value in loss_storage.items():
                        file.write(f"step={key} | loss={value}")
                    file.write(f"Model parameters saved to {model_path} at step {step}\n")
                loss_storage = {}
        print(f"step={step} | loss={loss_accum}")
        if step % 10000 == 9999 and run_validation:
            validation(model, val_loader, device, RunConfig, log_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to {model_path}")

def validation(model, val_loader, device, RunConfig, log_path):
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
            x_policy = model(board_state_tensor, special_token_tensor)
            loss = F.cross_entropy(x_policy, target_p_tensor)
            losses_list.append(loss.item())
        loss_accum = sum(losses_list)/len(losses_list)
    if log_path is not None:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Validation Loss: loss={loss_accum}\n")

    print(f"Validation Loss: loss={loss_accum}")

if run_training:
    training(model, train_loader, val_loader, optimizer, grad_accum_steps, device, RunConfig, model_path, log_path, VariableRunConfig.train_type, VariableRunConfig.masking)

if run_validation:
    validation(model, val_loader, device, RunConfig, log_path)


# if __name__ == '__main__':
#     main()

