import torch

with open("state_dict_1.txt", "r") as file:
    keys_1 = file.read().splitlines()

with open("state_dict_2.txt", "r") as file:
    keys_2 = file.read().splitlines()

unique_1 = []
unique_2 = []

for key in keys_1:
    if key not in keys_2:
        unique_1.append(key)
for key in keys_2:
    if key not in keys_1:
        unique_2.append(key)
    
print(unique_1)
print(unique_2)