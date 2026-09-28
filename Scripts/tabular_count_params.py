# %%
import torch
import torch.nn as nn
from tabular_train import TabularTransformer

def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_from_checkpoint():
    # Load the checkpoint
    checkpoint = torch.load('tabular_checkpoints/best_tabular.pth', 
                           map_location=torch.device('cpu'), weights_only=False)
    
    # Get the hyperparameters from the saved checkpoint
    hp = checkpoint['hyperparameters']
    
    # Create a new model instance with the same architecture
    model = TabularTransformer(
        attn_dropout=hp['attention_dropout'],
        ff_dropout=hp['feedforward_dropout'],
        dim=hp['dimensions'],
        depth=hp['depth'],
        heads=hp['heads']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"Tabular Transformer")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print detailed parameter information
    print("\nDetailed parameter breakdown:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters {'(trainable)' if param.requires_grad else '(frozen)'}")

if __name__ == "__main__":
    count_from_checkpoint()
    
# %%