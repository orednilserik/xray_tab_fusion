# %%
import torch

def main():
    # Load a specific fold's hyperparameters
    fold_number = 1  # Change this to the fold you're interested in
    checkpoint_path = f"fusion_pipeline/fold_{fold_number}/best_tabular_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    # Access the hyperparameters
    hyperparameters = checkpoint['hyperparameters']
    print(f"Fold {fold_number} hyperparameters:")
    for param_name, param_value in hyperparameters.items():
        print(f"  {param_name}: {param_value}")

if __name__ == "__main__":
    main()
# %%