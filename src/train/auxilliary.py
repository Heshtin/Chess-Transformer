def retrieve_iteration_number(write):
    iteration = 0
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
    return iteration

    
def write_to_hyperparam(log_path, total_params, HyperParamConfig, VariableRunConfig, DataConfig):
    with open(log_path, 'a') as log_file:
        log_file.write(f"Hyperparameters:\n")
        log_file.write(f"encoding scheme: {HyperParamConfig.encoding_scheme}")
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
        log_file.write(f"total no of parameters: {total_params}\n\n")
        log_file.write(f"Run Configuration: \n")
        log_file.write(f"train steps: {VariableRunConfig.train_steps}")
        log_file.write(f"masking: {VariableRunConfig.masking}")
        log_file.write(f"gpu_batch_size: {VariableRunConfig.gpu_batch_size}")
        log_file.write(f"n1, n2: {DataConfig.n1}, {DataConfig.n2}")



