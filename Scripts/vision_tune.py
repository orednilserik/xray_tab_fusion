import torch
import torch.nn as nn
import torch.optim as optim
from vision_preprocess import *
import os
import glob
import timm
import ray
import json
import time
from ray import tune
from ray.tune import CLIReporter, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import time
import gc
temp_ray_dir = "/home/mikmo937/Desktop/DATA2"
os.makedirs(temp_ray_dir, exist_ok=True)
os.environ["RAY_TMPDIR"] = temp_ray_dir
os.environ["TMPDIR"] = temp_ray_dir
ray_results_dir = temp_ray_dir + "/ray_results"
ray_storage_dir = temp_ray_dir + "/ray_storage"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters for tuning
pretrained_lr = 1e-5
finetuning_lr = 1e-4
weight_decay = 1e-6
patience = 30
num_epochs = 100

# Define vision transformer model for hyperparameter tuning
class VisionTransformer(nn.Module):
    def __init__(self, model_name='tiny_vit_5m_224.dist_in22k_ft_in1k', num_classes=1):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Output single unit logit
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
         
        embeddings = self.model.forward_features(x) # get features from the transformer
        embeddings = embeddings.mean(dim=[2, 3]) # average pooling of height and width to match the expected input dimensions
        preds = self.model.forward_head(embeddings)  # output the logits for predictions
        
        # output embeddings and predictions
        return embeddings, preds


def tune_model(config, train_loader, val_loader, class_weights):
    # Prepare data with augmentations
    #transform = previous_augmentations()
    #images = ImgData(source_path, transform=transform)

    # Load model
    model = VisionTransformer().to(device)
    
    # Get class weights
    #class_weights = images.class_weights
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Create parameter groups
    pretrained_params = [param for name, param in model.named_parameters() if 'head' not in name]
    finetuning_params = [param for name, param in model.named_parameters() if 'head' in name]
    
    # Configure optimizer with tuned learning rates
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': config["pretrained_lr"]},
        {'params': finetuning_params, 'lr': config["finetuning_lr"]}
    ], weight_decay=config["weight_decay"])

    # Load checkpoint if available
    checkpoint = get_checkpoint()
    if checkpoint:
        checkpoint_dict = checkpoint.as_directory()
        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    
    # Track best validation accuracy for early stopping
    best_val_accuracy = 0
    early_stop_counter = 0
    patience = config.get("patience", 10)
    
    # Training loop
    for epoch in range(config.get("epochs", 25)):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for patient_id, images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs[1]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach() * images.size(0)
            predicted = (torch.sigmoid(logits) > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().detach()
        
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for patient_id, images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1)
                
                outputs = model(images)
                logits = outputs[1]
                loss = criterion(logits, labels)
                val_loss += loss.detach() * images.size(0)
                
                predicted = (torch.sigmoid(logits) > 0.5).long()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().detach()
        
        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total

        # Report metrics to Ray Tune
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        tune.report({
            "train_accuracy": train_accuracy.item(),
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "val_accuracy": val_accuracy.item()
        })

        # Save best model and check for early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
            
            # Save checkpoint for best model
            path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            break


def run_hyperparameter_tuning(train_loader=None, val_loader=None, class_weights=None, fold_no=None, cv = True):
    
    if not cv:
        # check if a tuned configuration already exists
        config_path = os.path.join("image_checkpoints", "best_config.json") if fold_no is None else os.path.join("fusion_pipeline", f"fold_{fold_no}", "best_config.json")
    else:
        # check if a tuned configuration already exists
        config_path = os.path.join("fusion_pipeline", "best_config_img.json") if fold_no is None else os.path.join("fusion_pipeline", f"fold_{fold_no}", "best_config_img.json")
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if os.path.exists(config_path):
        print("Loading existing best hyperparameter configuration")
        with open(config_path, 'r') as f:
            best_config = json.load(f)
        return best_config
    
    # making sure raytune is not initialized already
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
            log_to_driver = False,
            ignore_reinit_error=True
        )
    
    
    
    config = {
        "pretrained_lr": tune.loguniform(1e-6, 1e-4),
        "finetuning_lr": tune.loguniform(1e-5, 5e-3),
        "weight_decay": tune.loguniform(1e-7, 1e-5),
        "batch_size": tune.choice([8, 16, 32]),
        "epochs": 25,
        "patience": 10
    }
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,  # max epochs
        grace_period=5,  # min epochs before pruning
        reduction_factor=2
    )
    
    wrapped_model = tune.with_parameters(tune_model,
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         class_weights=class_weights)
    """
    tuning_path = os.path.join("image_checkpoints", "ray_results")
    if fold_no is not None:
        tuning_path = os.path.join("image_checkpoints", f"ray_results_fold_{fold_no}")
    
    os.makedirs(tuning_path, exist_ok=True)
    """
    # Initialize Ray Tune
    start_time = time.time()
    result = tune.run(
        wrapped_model,
        resources_per_trial={"cpu": 12, "gpu": 0.5},  # 4 cores and half GPU per trial
        config=config,
        num_samples=100,  # number of trials to run
        scheduler=scheduler,
        #storage_path = tuning_path,
        keep_checkpoints_num=1,
        verbose = 0,
        checkpoint_score_attr="val_loss",
        progress_reporter=CLIReporter(
            metric_columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"])
    )
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Get best trial and configuration
    best_trial = result.get_best_trial("val_loss", "min", "last")
    best_config = best_trial.config
    best_accuracy = best_trial.last_result["val_accuracy"]
    
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation accuracy: {best_accuracy}")
    
    with open(config_path, 'w') as f:
        json.dump(best_config, f)
    
    # Update global hyperparameters with best found values
    global pretrained_lr, finetuning_lr, weight_decay
    pretrained_lr = best_config["pretrained_lr"]
    finetuning_lr = best_config["finetuning_lr"]
    weight_decay = best_config["weight_decay"]
    
    return best_config
    


if __name__ == "__main__":
    best_config = run_hyperparameter_tuning()
    print("Hyperparameter tuning completed.")
    print(f"Best configuration: {best_config}")
# %%