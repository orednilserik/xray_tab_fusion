# %%
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import ray
import json
import time
from ray import tune
from ray.tune import CLIReporter, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import gc
from fusion_conv_late_mc import LateFusion, ProjectionLayer, compute_class_weights
from extract_embeddings import extract_embeddings

# Set up Ray temp directory
temp_ray_dir = "/home/mikmo937/Desktop/DATA2"
os.makedirs(temp_ray_dir, exist_ok=True)
os.environ["RAY_TMPDIR"] = temp_ray_dir
os.environ["TMPDIR"] = temp_ray_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters
DEFAULT_CONFIG = {
    "fused_dim": 128,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "weight_decay": 1e-3
}

def prepare_dataloaders(train_data, val_data, batch_size):
    """Prepare dataloaders for training and validation"""
    # Extract labels and patient IDs
    train_labels_dict = {item.pid: item.label for item in train_data}
    val_labels_dict = {item.pid: item.label for item in val_data}
    
    # Move to CPU first to avoid any GPU->NumPy conversions
    train_tab_emb = torch.stack([item.tabular.cpu() if torch.is_tensor(item.tabular) and item.tabular.is_cuda else item.tabular for item in train_data])
    train_img_emb = torch.stack([item.vision.cpu() if torch.is_tensor(item.vision) and item.vision.is_cuda else item.vision for item in train_data])
    train_patient_ids = torch.tensor([item.pid for item in train_data]).long()
    
    val_tab_emb = torch.stack([item.tabular.cpu() if torch.is_tensor(item.tabular) and item.tabular.is_cuda else item.tabular for item in val_data])
    val_img_emb = torch.stack([item.vision.cpu() if torch.is_tensor(item.vision) and item.vision.is_cuda else item.vision for item in val_data])
    val_patient_ids = torch.tensor([item.pid for item in val_data]).long()
    
    # Extract labels
    train_labels = torch.tensor([train_labels_dict[pid.item()] for pid in train_patient_ids]).float().view(-1, 1)
    val_labels = torch.tensor([val_labels_dict[pid.item()] for pid in val_patient_ids]).float().view(-1, 1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_tab_emb, train_img_emb, train_labels, train_patient_ids)
    val_dataset = TensorDataset(val_tab_emb, val_img_emb, val_labels, val_patient_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_labels

def tune_fusion_model(config, train_data=None, val_data=None):
    """Training function for Ray Tune"""
    # Prepare dataloaders with the config batch size
    train_loader, val_loader, train_labels = prepare_dataloaders(
        train_data, val_data, config["batch_size"]
    )
    
    first_item = train_data[0]
    vision_dim = first_item.vision.shape[0]
    tabular_dim = first_item.tabular.shape[0]
    
    # Calculate class weights
    pos_weight = compute_class_weights(train_labels)
    
    # Initialize model with hyperparameters from config
    model = LateFusion(
        num_classes=1,
        vision_dim=vision_dim,  # Make sure these match your embedding dimensions
        tabular_dim=tabular_dim,  # Make sure these match your embedding dimensions
        fused_dim=config["fused_dim"]
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config["learning_rate"], 
                           weight_decay=config["weight_decay"])
    
    # Load checkpoint if available
    checkpoint = get_checkpoint()
    if checkpoint:
        checkpoint_dict = checkpoint.as_directory()
        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    
    # Track metrics
    best_val_loss = float('inf')
    patience = config.get("patience", 10)
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(config.get("epochs", 50)):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_labels = []
        all_train_probs = []
        
        for tab_emb, img_emb, labels, _ in train_loader:
            tab_emb = tab_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(tab_emb, img_emb)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item() * labels.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Store for AUC calculation
            all_train_labels.extend(labels.detach().cpu().numpy().flatten())
            all_train_probs.extend(outputs.detach().cpu().numpy().flatten())
        
        # Calculate epoch metrics
        train_loss /= train_total
        train_accuracy = 100 * train_correct / train_total
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for tab_emb, img_emb, labels, _ in val_loader:
                tab_emb = tab_emb.to(device)
                img_emb = img_emb.to(device)
                labels = labels.to(device)
                
                outputs = model(tab_emb, img_emb)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_labels.extend(labels.detach().cpu().numpy().flatten())
                all_val_probs.extend(outputs.detach().cpu().numpy().flatten())
        
        # Calculate validation metrics
        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        
        # Report metrics to Ray Tune
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        tune.report({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_auc": val_auc
        })

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save checkpoint for best model
            path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            break

def run_hyperparameter_tuning(train_data=None, val_data=None, fold_no=None):
    """Run hyperparameter tuning with Ray Tune"""
    # Check if a tuned configuration already exists
    config_path = os.path.join("late_fusion_checkpoints", "best_config.json") if fold_no is None else os.path.join("late_fusion_checkpoints", f"fold_{fold_no}", "best_config.json")
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if os.path.exists(config_path):
        print("Loading existing best hyperparameter configuration")
        with open(config_path, 'r') as f:
            best_config = json.load(f)
        return best_config
    
    # Make sure ray is not already initialized
    ray_initialized = False
    gc.collect()
    ray.shutdown()
    torch.cuda.empty_cache()
    try:
        if ray.is_initialized():
            ray_initialized = True
    except:
        pass
    
    if not ray_initialized:
        ray.init(
            _temp_dir=temp_ray_dir,
            include_dashboard=False,
            log_to_driver=False,
            ignore_reinit_error=True
        )
    
    # Define search space
    config = {
        "fused_dim": tune.choice([64, 128, 256, 512]),
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "dropout_rate": 0.5,
        "epochs": 50,
        "patience": 15
    }
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_loss",  # Using AUC as the metric to optimize
        mode="min",
        max_t=50,  # max epochs
        grace_period=10,  # min epochs before pruning
        reduction_factor=2
    )
    
    wrapped_model = tune.with_parameters(
        tune_fusion_model,
        train_data=train_data,
        val_data=val_data
    )
    
    # Initialize Ray Tune
    start_time = time.time()
    result = tune.run(
        wrapped_model,
        resources_per_trial={"cpu": 8, "gpu": 0.5},  # Adjust based on your hardware
        config=config,
        num_samples=100,  # Number of trials to run
        scheduler=scheduler,
        keep_checkpoints_num=1,
        verbose=0,
        checkpoint_score_attr="val_loss",
        progress_reporter=CLIReporter(
            metric_columns=["train_loss", "train_accuracy", "train_auc", 
                          "val_loss", "val_accuracy", "val_auc"])
    )
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Get best trial and configuration
    best_trial = result.get_best_trial("val_loss", "min", "last")
    best_config = best_trial.config
    best_auc = best_trial.last_result["val_auc"]
    
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation AUC: {best_auc:.4f}")
    
    with open(config_path, 'w') as f:
        json.dump(best_config, f)
    
    return best_config

def train_with_best_config(config, train_data, val_data, fold_no):
    
    first_item = train_data[0]
    vision_dim = first_item.vision.shape[0]
    tabular_dim = first_item.tabular.shape[0]
    
    # Initialize model with best hyperparameters
    model = LateFusion(
        num_classes=1,
        vision_dim=vision_dim,
        tabular_dim=tabular_dim,
        fused_dim=config["fused_dim"]
    ).to(device)
    
    # Apply dropout rate from config to all dropout layers
    if "dropout_rate" in config:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = config["dropout_rate"]
    
    # Train using the original training function with best hyperparameters
    from fusion_conv_late_mc import train_fusion_model
    trained_model = train_fusion_model(
        model,
        train_data,
        val_data,
        num_epochs=200,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fold_no=fold_no,
        verbose=False
    )
    
    return trained_model

if __name__ == "__main__":
    
    folds = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
    
    for fold in folds:
        fold_no = int(fold.split('_')[1])  # extract the fold number
        pair_train_emb, pair_val_emb, pair_test_emb = extract_embeddings(fold)
    
   
        # Run hyperparameter tuning
        best_config = run_hyperparameter_tuning(pair_train_emb, pair_val_emb, fold_no)
        print(f"Hyperparameter tuning completed {fold}.")
        print(f"Best configuration: {best_config}")
    
        # Train with best configuration
        trained_model = train_with_best_config(best_config, pair_train_emb, pair_val_emb, fold_no)
        print("Model training completed with best hyperparameters.")
    
        from fusion_conv_late_mc import test_fusion_mc, aggregate_patient_results_mc
    
        test_results_mc_df = test_fusion_mc(trained_model, pair_test_emb, num_samples=100,fold_no=fold_no)
    
        patient_level_results_mc = aggregate_patient_results_mc(test_results_mc_df,fold_no=fold_no)
    
        print("Evaluation completed.")
# %%