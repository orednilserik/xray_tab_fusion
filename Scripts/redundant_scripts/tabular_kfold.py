# %%
#from tabular_preprocess import preprocess_data, create_dataloaders
from tabular_train import *
from redundant_scripts.tabular_test import test_model
#from extract_patients import load_patient_ID

import json
import numpy as np
from sklearn.model_selection import KFold
import torch.multiprocessing as mp
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_class_weights(labels_tensor):
    # make sure tensor is on CPU before converting to numpy
    if torch.is_tensor(labels_tensor):
        labels_tensor = labels_tensor.cpu().detach().numpy()
    else:
        # If it's already a numpy array or list, just use it
        labels_tensor = np.array(labels_tensor)
    
    labels = np.array(labels_tensor).flatten()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels)
    pos_weight = torch.tensor(class_weights.max() / class_weights.min(), dtype=torch.float32, device=device)
    return pos_weight


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

def train_single_fold(fold_no, fold_data, config = None, fusion = True, verbose = False):

    print(f"Training fold {fold_no}")
    
    # data for folds
    current_fold = fold_data[fold_no]
    train_patients = current_fold["train"]
    val_patients = current_fold["val"]
    test_patients = current_fold["test"]
    
    # load data
    full_data = preprocess_data()
    train_indices = [i for i in range(len(full_data)) if full_data.iloc[i]['rlpnr'] in train_patients]
    val_indices = [i for i in range(len(full_data)) if full_data.iloc[i]['rlpnr'] in val_patients]
    test_indices = [i for i in range(len(full_data)) if full_data.iloc[i]['rlpnr'] in test_patients]
    
    # create subsets for PyTorch compatibility
    train_dataset, val_dataset, test_dataset = create_subset(
        full_data, 
        train_indices=train_indices,
        val_indices=val_indices, 
        test_indices=test_indices
    )
    # data loaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    
    # hyperparams from config
    if config is None:
        best_config = run_hyperparameter_tuning()
    else:
        best_config = config
    
    # initialize model
    model = TabularTransformer(
        attn_dropout=best_config["attention_dropout"],
        ff_dropout=best_config["feedforward_dropout"],
        dim=best_config["dimensions"],
        depth=best_config["depth"],
        heads=best_config["heads"]
    ).to(device)
    
    # training params
    class_weights = compute_class_weights(full_data['stressfx_Y'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.AdamW(model.parameters(),
                            lr=best_config["learning_rate"],
                            weight_decay=best_config["weight_decay"])
    
    fold_path = f"tabular_checkpoints/temp_fold_{fold_no}.pth"
    
    # train model
    train_model(model = model,
                train_loader = train_loader,
                val_loader = val_loader,
                criterion = criterion,
                optimizer = optimizer,
                best_config = best_config,
                num_epochs = 150,
                checkpoint_path = fold_path,
                verbose = verbose)
    
    training_checkpoint = torch.load(fold_path, weights_only=False)
    print(f"Training checkpoint keys: {training_checkpoint.keys()}")
    val_loss = training_checkpoint['val_loss']
    
    test_path = fold_path
    test_metrics = test_model(model,
                              test_loader,
                              criterion,
                              checkpoint_path=test_path,
                              verbose = verbose)                  
    
    # evaluate on test and save results for particular fold
    if not fusion:    
        fold_results = {
            'val_loss': val_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'checkpoint_path': fold_path,
            'hyperparameters': best_config
        }
    
    else: # fusion is True
        fold_results = {
            'val_loss': val_loss,
            'checkpoint_path': fold_path,
            'hyperparameters': best_config
        }
        
    return fold_results

def run_kfold_cv(n_splits = 5, test_frac = 0.1, seed = 181, fusion = True, verbose = False):
    """run k-fold cross-validation and return the best model based on validation loss"""
    print(f"Running {n_splits}-fold cross-validation")
    
    folds = kfold_splits(n_splits, test_frac, seed)
    
    fold_results = {} # dict to save results from k cv
    
    for fold_no in range(1, n_splits+1):
        print(f"Running fold {fold_no}/{n_splits}")
            
        # train and evaluate
        fold_result = train_single_fold(fold_no, folds, None, fusion, verbose)
        fold_results[fold_no] = fold_result
    
    val_losses = [fold['val_loss'].cpu() if torch.is_tensor(fold['val_loss']) else fold['val_loss'] 
                  for fold in fold_results.values()]
    
    best_fold_no = np.argmin(val_losses) + 1
    best_fold = fold_results[best_fold_no]
    
    os.makedirs("tabular_checkpoints", exist_ok=True)
    best_model_path = "tabular_checkpoints/best_tabular_cv.pth"
    
    # summary statistics for when not doing fusion
    if not fusion:
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        summary = {}
        for metric in metrics:
            values = [fold[metric] for fold in fold_results.values()]
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
        
        print("\nK-fold Cross-Validation Summary:")
        for metric in metrics:
            print(f"Mean {metric}: {summary[f'mean_{metric}']:.4f} ± {summary[f'std_{metric}']:.4f}")
    
    print(f"Best fold: {best_fold_no} with validation loss: {best_fold['val_loss']:.4f}")
    
    # save best model
    best_checkpoint = torch.load(best_fold['checkpoint_path'], weights_only=False)
    torch.save(best_checkpoint, best_model_path)

    return fold_results, best_fold


def main():
    start_time = time.time()
    
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    all_results, best_fold = run_kfold_cv(
        n_splits=5,
        seed=181,
        test_frac=0.1,
        fusion = False,
        verbose = True
    )

    end_time = time.time()
    print(f"Total K-fold cross-validation time: {end_time - start_time:.2f} seconds")
    print(f"Best fold results: {best_fold}")
    
if __name__ == "__main__":
    main()
# %%