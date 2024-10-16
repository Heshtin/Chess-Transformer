import torch

state_dict_path = "/workspace/runs/bay_optim_run_1/iters2/state_dict_v18.pth"

print("loading")
# Load the state_dict from the .pth file
state_dict = torch.load(state_dict_path)
if state_dict is not None:
    print("state_dict is not empty")
# Write the keys to a file, one per line
with open("state_dict_2.txt", "a") as file:
    print("writing")
    for key in state_dict.keys():
        file.write(key + "\n")
