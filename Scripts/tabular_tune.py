import torch
import torch.nn as nn
import torch.optim as optim
from tab_transformer_pytorch import FTTransformer
from tabular_preprocess import *
import os
from sklearn.metrics import precision_recall_fscore_support as score
from ray import tune
import ray
from ray.tune import CLIReporter, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 45 # amount of variables put into model
num_classes = 1 # binary classification

# default hyperparameters for tuning
attention_dropout = 0.2     # default 0.1
feedforward_dropout = 0.2   # default 0.1
dimensions = 32             # default 32
depth = 6                   # default 6
heads = 8                   # default 8
learning_rate = 1e-5
weight_decay = 1e-6
number_of_epochs = 50

# define tabular transformer model
class TabularTransformer(nn.Module):
    def __init__(self, attn_dropout=attention_dropout, ff_dropout=feedforward_dropout, 
                 dim=dimensions, depth=depth, heads=heads):
        super(TabularTransformer, self).__init__()
        self.model = FTTransformer(
            categories=(2,),              # empty tuple, since there are no categorical variables
            num_continuous=input_size-1,  # number of continuous variables
            dim=dim,             
            dim_out=dim,         # output dimensions for manual forward pass
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        
        # separate classifier head for binary prediction
        self.classifier = nn.Linear(dim, num_classes)
            
    def forward(self, x):
        categorical_data = x[:, 0].unsqueeze(1).long() # categorical data
        continuous_data = x[:, 1:] # continuous data
        
        embeddings = self.model(categorical_data, continuous_data) # get features from the transformer
        preds = self.classifier(embeddings) # output the logits for predictions
                
        return embeddings, preds

def tune_model(config):
    # load and prepare data
    final_df = preprocess_data()
    train_data, val_data, test_data = create_subset(final_df)
    train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=config["batch_size"])

    # load model with hyperparameters
    model = TabularTransformer(
        attn_dropout=config["attention_dropout"],
        ff_dropout=config["feedforward_dropout"],
        dim=config.get("dimensions", dimensions),
        depth=config.get("depth", depth),
        heads=config.get("heads", heads)
    ).to(device)

    # optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    checkpoint = get_checkpoint()
    if checkpoint:
        checkpoint_dict = checkpoint.as_directory()
        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    # class weights for loss function
    class_weights = compute_class_weights(train_data)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
     # training loop for tuning
    best_val_accuracy = 0
    early_stop_counter = 0
    patience = config.get("patience", 10)
    
    for epoch in range(config.get("epochs", 25)):  # fewer epochs for tuning than regular training
        # initialize training same as normal training loop
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for continuous_data, labels, patient_id in train_loader:
            continuous_data = continuous_data.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(continuous_data)
            logits = outputs[1]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach() * continuous_data.size(0)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().detach()
        
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        
        # initialize validation same as normal validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for continuous_data, labels, patient_id in val_loader:
                continuous_data = continuous_data.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float().view(-1, 1)
                
                outputs = model(continuous_data)
                logits = outputs[1]
                loss = criterion(logits, labels)
                val_loss += loss.detach() * continuous_data.size(0)
                
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().detach()
        
        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total

        # report metrics, not saving model until best is found
        checkpoint_dir = "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        tune.report({
            "train_accuracy": train_accuracy.item(),
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "val_accuracy": val_accuracy.item()
        }) 

        # save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        
        # simple early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            break
        
def run_hyperparameter_tuning(fold_no=None, cv = True):
    # making sure raytune is not initialized already
    ray_initialized = False
    ray.shutdown()
    try:
        if ray.is_initialized():
            ray_initialized = True
    except:
        pass
    
    if not ray_initialized:
        ray.init(include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    
    if not cv:
        # check if a tuned configuration already exists
        config_path = os.path.join("tabular_checkpoints", "best_config.json") if fold_no is None else os.path.join("fusion_pipeline", f"fold_{fold_no}", "best_config.json")
    else:
        # check if a tuned configuration already exists
        config_path = os.path.join("fusion_pipeline", "best_config_tab.json") if fold_no is None else os.path.join("fusion_pipeline", f"fold_{fold_no}", "best_config_tab.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # if it exists, load that instead of running new tuning
    if os.path.exists(config_path):
        print("Loading existing best hyperparameter configuration")
        with open(config_path, 'r') as f:
            best_config = json.load(f)
        return best_config
    
    
    # search space for hyperparameters
    config = {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "weight_decay": tune.loguniform(1e-7, 1e-5),
        "attention_dropout": tune.uniform(0.1, 0.5),
        "feedforward_dropout": tune.uniform(0.1, 0.5),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "dimensions": tune.choice([16, 32, 64]),
        "depth": tune.choice([4, 6, 8]),
        "heads": tune.choice([4, 8]),
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
    
    # initialize RayTune
    result = tune.run(
        tune_model,
        resources_per_trial={"cpu": 8, "gpu": 0.5},  # utilize 8 cores and half of GPU per trial
        config=config,
        num_samples=100,  # trials to run
        scheduler=scheduler,
        verbose = 0,
        progress_reporter=CLIReporter(
        metric_columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"])
    )
    
    # get best trial and configuration
    best_trial = result.get_best_trial("val_loss", "min", "last")
    best_config = best_trial.config
    best_accuracy = best_trial.last_result["val_accuracy"]
    
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation accuracy: {best_accuracy}")
    
    with open(config_path, 'w') as f:
            json.dump(best_config, f)
    
    # update global hyperparameters with best found values
    global attention_dropout, feedforward_dropout, learning_rate, weight_decay, dimensions, depth, heads
    attention_dropout = best_config["attention_dropout"]
    feedforward_dropout = best_config["feedforward_dropout"]
    learning_rate = best_config["learning_rate"]
    weight_decay = best_config["weight_decay"]
    dimensions = best_config["dimensions"]
    depth = best_config["depth"]
    heads = best_config["heads"]
    
    return best_config