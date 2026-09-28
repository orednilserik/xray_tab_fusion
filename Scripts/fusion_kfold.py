# %%
from vision_train import train_model as train_model_img, VisionTransformer
from vision_tune import run_hyperparameter_tuning as tune_hyperparams_img
from vision_test_mc import test_model as test_model_img_mc
from vision_preprocess import previous_augmentations, ImgData, create_subset as create_subset_img, create_dataloaders
from tabular_tune import run_hyperparameter_tuning as tune_hyperparams_tab
from tabular_train import train_model as train_model_tab, TabularTransformer
from tabular_test_mc import test_model as test_model_tab_mc
from tabular_preprocess import preprocess_data, create_subset as create_subset_tab, compute_class_weights
from extract_patients import load_patient_ID, source_path
from tabular_train import extracted_attn_weights

import gc
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
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kfold_splits(n_splits = 5, test_frac = 0.1, seed = 181):
    patient_ids = load_patient_ID()

    if seed is not None:
        random.seed(seed) # random seed for reproducibility
        random.shuffle(patient_ids) # random shuffle
    else:
        random.shuffle(patient_ids)
    
    total_patients = len(patient_ids)
    test_size = math.ceil(total_patients * test_frac)
    
    test_patients = patient_ids[:test_size] # remove set of patients to use as test
    train_val_patients = patient_ids[test_size:] # only performing split on train/val
    
    # initialize k-fold
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = seed)
    
    # dict for storing information
    fold_data = {}
    
    # generate folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_patients)):
        train_patients = [train_val_patients[idx] for idx in train_idx]
        val_patients = [train_val_patients[idx] for idx in val_idx]
        
        fold_data[fold_idx+1] = {"train": train_patients, 
                                 "val": val_patients,
                                 "test": test_patients}
        

    return fold_data

def train_single_fold(fold_no, fold_data, fusion=True, images=None, tab_config=None, img_config=None):
     # get fold data
    current_fold = fold_data[fold_no]
    train_patients = current_fold["train"]
    val_patients = current_fold["val"]
    test_patients = current_fold["test"]
    
    # load images if not provided
    if images is None:
        # load data and preprocessing
        transform = previous_augmentations()
        images = ImgData(source_path, transform=transform)
    
    # load tabular data
    full_data = preprocess_data()
    train_indices = [i for i in range(len(full_data)) if full_data.iloc[i]['rlpnr'] in train_patients]
    val_indices = [i for i in range(len(full_data)) if full_data.iloc[i]['rlpnr'] in val_patients]
    test_indices = [i for i in range(len(full_data)) if full_data.iloc[i]['rlpnr'] in test_patients]
    
    # subsets for each modality
    train_data_tab, val_data_tab, test_data_tab = create_subset_tab(full_data, train_indices, val_indices, test_indices)
    train_data_img, val_data_img, test_data_img = create_subset_img(images, train_patients, val_patients, test_patients)
    
    
    # data loaders for each modality
    train_load_tab, val_load_tab, test_load_tab = create_dataloaders(train_data_tab, val_data_tab, test_data_tab)
    train_load_img, val_load_img, test_load_img = create_dataloaders(train_data_img, val_data_img, test_data_img)
    
    # class weights from tabular data
    class_weights = compute_class_weights(train_data_tab)
    
    # look for already tuned hyperparameters
    if tab_config is None:
        best_config_tab = tune_hyperparams_tab(fold_no=fold_no)
    else:
        best_config_tab = tab_config
    torch.cuda.empty_cache()
    if img_config is None:
        best_config_img = tune_hyperparams_img(train_loader = train_load_img,
                                               val_loader = val_load_img,
                                               class_weights = class_weights,
                                               fold_no = fold_no)
                                                
                                                
                                            
    else:
        best_config_img = img_config
    
    # recreating data loaders if tuned batch size differs
    init_batch_size = 16
    if best_config_tab.get("batch_size", init_batch_size) != init_batch_size:
        train_load_tab, val_load_tab, test_load_tab = create_dataloaders(train_data_tab, 
                                                                        val_data_tab, 
                                                                        test_data_tab, 
                                                                        best_config_tab["batch_size"])
        
    if best_config_img.get("batch_size", init_batch_size) != init_batch_size:
        train_load_img, val_load_img, test_load_img = create_dataloaders(train_data_img, 
                                                                        val_data_img, 
                                                                        test_data_img, 
                                                                        best_config_img["batch_size"])
    
    
    # initialize models
    tab_model = TabularTransformer(
        attn_dropout=best_config_tab["attention_dropout"],
        ff_dropout=best_config_tab["feedforward_dropout"],
        dim=best_config_tab["dimensions"],
        depth=best_config_tab["depth"],
        heads=best_config_tab["heads"]
    ).to(device)
    
    
    img_model = VisionTransformer(num_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) # criterion is same for both models
    
    # optimizer for tabular data 
    tab_optimizer = optim.AdamW(tab_model.parameters(),
                                lr=best_config_tab["learning_rate"],
                                weight_decay=best_config_tab["weight_decay"])
    
    # only relevant for img model
    pretrained_params = [param for name, param in img_model.named_parameters() if 'head' not in name]
    finetuning_params = [param for name, param in img_model.named_parameters() if 'head' in name]
    
    # optimizer for img data
    img_optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': best_config_img["pretrained_lr"]},
        {'params': finetuning_params, 'lr': best_config_img["finetuning_lr"]}
    ], weight_decay=best_config_img["weight_decay"])
    
    # create checkpoints for folds
    os.makedirs(os.path.join("fusion_pipeline", f"fold_{fold_no}"), exist_ok=True)
    fold_path_img = os.path.join("fusion_pipeline", f"fold_{fold_no}", "best_vision_model.pth")
    fold_path_tab = os.path.join("fusion_pipeline", f"fold_{fold_no}", "best_tabular_model.pth")
    
    # train vision model for specific fold
    train_model_img(model = img_model,
                    train_loader = train_load_img,
                    val_loader = val_load_img,
                    criterion = criterion,
                    optimizer = img_optimizer,
                    best_config = best_config_img,
                    num_epochs = 150,
                    patience_threshold = 50,
                    verbose = False,
                    checkpoint_path=fold_path_img)

    # load the best vision model and its validation loss
    img_train_checkpoint = torch.load(fold_path_img, weights_only=False)
    val_loss_img = img_train_checkpoint["loss"]
    torch.cuda.empty_cache()
    extracted_attn_weights.clear()
    # train tabular model for specific fold
    train_model_tab(model = tab_model,
                    train_loader = train_load_tab,
                    val_loader = val_load_tab,
                    criterion = criterion,
                    optimizer = tab_optimizer,
                    best_config = best_config_tab,
                    num_epochs = 80,
                    checkpoint_path = fold_path_tab,
                    verbose = False)
    
    # load the best tabular model and its validation loss
    tab_train_checkpoint = torch.load(fold_path_tab, weights_only=False)
    val_loss_tab = tab_train_checkpoint['val_loss']
    torch.cuda.empty_cache()
    
    # paths for test data
    test_path_img = fold_path_img
    test_path_tab = fold_path_tab
    
    # call test models to generate embeddings
    test_metrics_img = test_model_img_mc(model = img_model,
                                      test_loader = test_load_img,
                                      criterion = criterion,
                                      fusion = False,
                                      verbose = False,
                                      checkpoint_path=test_path_img)
    
    test_metrics_tab = test_model_tab_mc(model = tab_model,
                                  test_loader = test_load_tab,
                                  criterion = criterion,
                                  checkpoint_path=test_path_tab,
                                  verbose = False)  
    
    # save feature importances
    #feat_imp = process_attention_weights(fold_no)
    
    
    # if not performing fusion, test metrics are of interest                 
    if not fusion:
        fold_results = {
            'val_loss_img': val_loss_img,
            'val_loss_tab': val_loss_tab,
            'test_accuracy_tab': test_metrics_tab['accuracy'],
            'test_precision_tab': test_metrics_tab['precision'],
            'test_recall_tab': test_metrics_tab['recall'],
            'test_f1_tab': test_metrics_tab['f1'],
            'test_accuracy_img': test_metrics_img['accuracy'],
            'test_precision_img': test_metrics_img['precision'],
            'test_recall_img': test_metrics_img['recall'],
            'test_f1_img': test_metrics_img['f1'],
            'checkpoint_path_img': fold_path_img,
            'checkpoint_path_tab': fold_path_tab,
            'hyperparameters_img': best_config_img,
            'hyperparameters_tab': best_config_tab
            #'feature_importance': feat_imp.cpu().tolist()
        }        
    
    # if running fusion, only val_loss and checkpoints are of interest
    else:
         fold_results = {
            'val_loss_img': val_loss_img,
            'val_loss_tab': val_loss_tab,
            'checkpoint_path_img': fold_path_img,
            'checkpoint_path_tab': fold_path_tab,
            'hyperparameters_img': best_config_img,
            'hyperparameters_tab': best_config_tab
            #'feature_importance': feat_imp.cpu().tolist()
        } 
                                                  
    torch.cuda.empty_cache()
    return fold_results                
                                        

def run_kfold_cv(n_splits = 5, test_frac = 0.1, seed = 181, fusion = True):
    """run k-fold cross-validation and return the best model based on validation loss"""
    print(f"Running {n_splits}-fold cross-validation")
    torch.cuda.empty_cache()
    gc.collect()
    # load images
    transform = previous_augmentations()
    images = ImgData(source_path, transform=transform)
    
    # perform kfold splits
    folds = kfold_splits(n_splits, test_frac, seed)
    
    fold_results = {} # dict to save results from k cv
    
    for fold_no in range(1, n_splits+1):
        print(f"Running fold {fold_no}/{n_splits}")
        torch.cuda.empty_cache()
        gc.collect()    
        # train and evaluate models
        fold_result = train_single_fold(fold_no = fold_no, 
                                        fold_data = folds, 
                                        fusion = fusion, 
                                        images = images,
                                        tab_config = None,
                                        img_config = None)
        
        fold_results[fold_no] = fold_result
    
    # find the best fold based on validation loss from image model
    val_losses = [fold['val_loss_img'].cpu() if torch.is_tensor(fold['val_loss_img']) else fold['val_loss_img'] 
                  for fold in fold_results.values()]
    best_fold_no = np.argmin(val_losses) + 1
    best_fold = fold_results[best_fold_no]
    
    os.makedirs("fusion_pipeline", exist_ok=True)
    best_tab_path = "fusion_pipeline/best_tabular.pth"
    best_img_path = "fusion_pipeline/best_vision.pth"
    
    # summary statistics for when not doing fusion
    if not fusion:
        metrics = ['test_accuracy_tab', 'test_precision_tab', 'test_recall_tab', 'test_f1_tab',
                   'test_accuracy_img', 'test_precision_img', 'test_recall_img', 'test_f1_img']
        summary = {}
        for metric in metrics:
            values = [fold[metric] for fold in fold_results.values()]
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
        
        print("\nK-fold Cross-Validation Summary:")
        for metric in metrics:
            print(f"Mean {metric}: {summary[f'mean_{metric}']:.4f} ± {summary[f'std_{metric}']:.4f}")
    
    print(f"Best fold: {best_fold_no} with validation loss: {best_fold['val_loss_img']:.4f}")
    
    best_checkpoint_img = torch.load(best_fold['checkpoint_path_img'], weights_only=False)
    best_checkpoint_tab = torch.load(best_fold['checkpoint_path_tab'], weights_only=False)
    
    torch.save(best_checkpoint_img, best_img_path)
    torch.save(best_checkpoint_tab, best_tab_path)
    
    return fold_results, best_fold
"""
# feature importance
def process_attention_weights(fold_no):
    if extracted_attn_weights:
        attns = extracted_attn_weights[-1]  # shape: [num_layers, batch, heads, tokens, tokens]
        # average across batch, heads, layers
        cls_attn = attns[:, :, :, 0, 1:]  # [layers, batch, heads, features]
        feature_importance = cls_attn.mean(dim=(0, 1, 2))
        
        full_data = preprocess_data()
        features = [c for c in full_data.columns if c not in ['rlpnr', 'stressfx_Y']]
        
        # Print feature importances
        imp_pairs = list(zip(features, feature_importance.tolist()))
        imp_pairs.sort(key=lambda x: x[1], reverse=True)  # sort by importance
        
        print(f"\nFeature Importances for Fold {fold_no}:")
        print("-" * 40)
        for name, importance in imp_pairs:
            print(f"Feature: {name:20s}, Importance: {importance:.4f}")
        
        # Save the feature importance for this fold
        os.makedirs(os.path.join("fusion_pipeline", f"fold_{fold_no}"), exist_ok=True)
        torch.save(feature_importance, os.path.join("fusion_pipeline", f"fold_{fold_no}", "feature_importance.pt"))
        
        
        return feature_importance
    return None

"""
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