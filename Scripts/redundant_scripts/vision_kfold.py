# %%
from vision_train import train_model, VisionTransformer
from vision_tune import run_hyperparameter_tuning
from redundant_scripts.vision_test import test_model
from redundant_scripts.tabular_kfold import kfold_splits
from extract_patients import load_patient_ID, source_path
from vision_preprocess import *
import json
import numpy as np
import random
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import torch.multiprocessing as mp
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_single_fold(fold_no, fold_data, config=None, fusion=True, images=None):
    print(f"Training fold {fold_no}")
          
    # get fold data
    current_fold = fold_data[fold_no]
    train_patients = current_fold["train"]
    val_patients = current_fold["val"]
    test_patients = current_fold["test"]
    
    if images is None:
        # load data and preprocessing
        transform = previous_augmentations()
        images = ImgData(source_path, transform=transform)
    
    # subsets
    train_dataset, val_dataset, test_dataset = create_subset(images, train_patients, val_patients, test_patients)
    
    # weights
    class_weights = images.class_weights
    
    # create dataloaders with initial batch size
    init_batch_size = 16
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, 
                                                               val_dataset, 
                                                               test_dataset, 
                                                               init_batch_size,
                                                               cv=True)
    
    # hyperparamter tuning
    if config is None:
        best_config = run_hyperparameter_tuning(train_loader = train_loader,
                                                val_loader = val_loader,
                                                class_weights = class_weights,
                                                fold_no = fold_no)
    else:
        best_config = config
        
    # recreating data loaders if tuned batch size differs
    if best_config.get("batch_size", init_batch_size) != init_batch_size:
        train_loader, val_loader, test_loader = create_dataloaders(train_dataset, 
                                                                   val_dataset, 
                                                                   test_dataset, 
                                                                   best_config["batch_size"],
                                                                   cv=True)
        
    # initialize model
    model = VisionTransformer(num_classes = 1).to(device)
    
    # loading hyperparameters
    pretrained_params = [param for name, param in model.named_parameters() if 'head' not in name]
    finetuning_params = [param for name, param in model.named_parameters() if 'head' in name]
    
    # optimizer using dynamic AdamW, dynamic lr, weight decay
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': best_config["pretrained_lr"]},
        {'params': finetuning_params, 'lr': best_config["finetuning_lr"]}
    ], weight_decay=best_config["weight_decay"])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # create checkpoints for folds
    os.makedirs(os.path.join("image_checkpoints", f"fold_{fold_no}"), exist_ok=True)
    fold_path = os.path.join("image_checkpoints", f"fold_{fold_no}", "best_model.pth")
    
    # train model for specific fold
    train_model(model = model,
                train_loader = train_loader,
                val_loader = val_loader,
                criterion = criterion,
                optimizer = optimizer,
                best_config = best_config,
                num_epochs = 200,
                patience_threshold = 50,
                verbose = False,
                checkpoint_path=fold_path)
    
    # load the best model and its validation loss
    training_checkpoint = torch.load(fold_path, weights_only=False)
    val_loss = training_checkpoint["loss"]
    
    if fusion: # test metrics not interesting
        fold_results = {
            'val_loss': val_loss,
            'checkpoint': fold_path,
            'hyperparameters': best_config
        }
    else:
        test_metrics = test_model(model = model,
                                 test_loader = test_loader,
                                 criterion = criterion,
                                 fusion = False,
                                 verbose = False,
                                 checkpoint_path=fold_path)

        fold_results = {
            'val_loss': val_loss,
            'checkpoint': fold_path,
            'hyperparameters': best_config,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        }

    torch.cuda.empty_cache()
    return fold_results

def run_kfold_cv(n_splits = 5, test_frac = 0.1, seed = 181, fusion = True):
    print(f"Running {n_splits}-fold CV")
    
    # load images
    transform = previous_augmentations()
    images = ImgData(source_path, transform=transform)
    
    folds = kfold_splits(n_splits = n_splits, test_frac = test_frac, seed = seed)
    fold_results = {} # dict to save results from k cv
    
    for fold_no in range(1, n_splits+1):
        print(f"Running fold {fold_no}/{n_splits}")
        
        # train and evaluate fold w.r.t validation loss
        fold_result = train_single_fold(fold_no, folds, images = images, fusion=fusion)
        fold_results[fold_no] = fold_result
        
    val_losses = [fold['val_loss'].cpu() if torch.is_tensor(fold['val_loss']) else fold['val_loss'] 
                  for fold in fold_results.values()]
    
    best_fold_no = np.argmin(val_losses) + 1
    best_fold = fold_results[best_fold_no]
    
    # path to best model
    os.makedirs("image_checkpoints", exist_ok=True)
    best_model_cv_path = "image_checkpoints/best_vision_cv.pth"
    
    # summary statistics on test when not doing fusion
    if not fusion:
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        summary = {}
        for metric in metrics:
            values = [fold[metric] for fold in fold_results.values()]
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
        
        print("\nVision K-fold Cross-Validation Summary:")
        for metric in metrics:
            print(f"Mean {metric}: {summary[f'mean_{metric}']:.4f} ± {summary[f'std_{metric}']:.4f}")
    
    print(f"Best vision fold: {best_fold_no} with validation loss: {best_fold['val_loss']:.4f}")
    
    # save best model
    best_checkpoint = torch.load(best_fold['checkpoint'], weights_only=False)
    torch.save(best_checkpoint, best_model_cv_path)
    
    return fold_results, best_fold

def main():
    start_time = time.time()
    
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    all_results, best_fold = run_kfold_cv(
        n_splits=5,
        seed=181,  # should be same as tabular model
        test_frac=0.1,
        fusion=False
    )
    
    end_time = time.time()
    print(f"Total K-fold cross-validation time: {end_time - start_time:.2f} seconds")
    print(f"Best fold results: {best_fold}")

if __name__ == "__main__":
    main()
# %%