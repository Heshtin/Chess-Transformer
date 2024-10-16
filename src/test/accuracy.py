from accuracy_model import Chess
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
                board_state_tensor, special_token_tensor, target_p_tensor = next(val_iter)
            except StopIteration:
                break
            board_state_tensor, special_token_tensor, target_p_tensor = board_state_tensor.to(device), special_token_tensor.to(device), target_p_tensor.to(device)

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
