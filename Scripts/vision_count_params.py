# %%
import torch
import torch.nn as nn
import timm
from vision_train import VisionTransformer

def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_from_checkpoint():
    # Load the checkpoint
    checkpoint = torch.load('image_checkpoints/best_vision.pth', 
                           map_location=torch.device('cpu'), weights_only=False)
    
    # Get the model name from the saved hyperparameters
    model_name = checkpoint['hyperparameters']['model_name']
    
    # Create a new model instance with the same architecture
    model = VisionTransformer(num_classes=1, model_name=model_name)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"Model: {model_name}")
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