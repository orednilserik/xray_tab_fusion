# %%
import torch
import torch.nn as nn
import torch.optim as optim
from tab_transformer_pytorch import FTTransformer
from tabular_preprocess import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_recall_fscore_support as score
import glob
import numpy as np
from torchsummary import summary
from tabular_tune import run_hyperparameter_tuning
import torch.multiprocessing as mp
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for feature importances
extracted_attn_weights = []

# HYPERPARAMETERS
input_size = 45 # amount of variables put into model
num_classes = 1 # binary classification
number_of_epochs = 200

# define tabular transformer model
class TabularTransformer(nn.Module):
    def __init__(self, attn_dropout=0.2, ff_dropout=0.2, dim=32, depth=6, heads=8):
        super(TabularTransformer, self).__init__()
        # for extracting later
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
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
        
        embeddings= self.model(categorical_data, continuous_data) # get features from the transformer
        # return_attn=True
        #extracted_attn_weights.append(attention.detach().cpu()) # store attention weights
        preds = self.classifier(embeddings) # output the logits for predictions
                
        return embeddings, preds

# define training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, best_config, num_epochs=64,checkpoint_path='tabular_checkpoints/best_tabular.pth',verbose=True):
    best_val_loss = float('inf') # threshold for validation loss
    best_train_results = []
    best_val_results = []
    best_train_embeddings = {}
    best_val_embeddings = {}
    os.makedirs('tabular_checkpoints', exist_ok=True) # folder for checkpoint
    
    # for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_aucs = []
    val_aucs = []
    train_mccs = []
    val_mccs = []
    
    for epoch in range(num_epochs):
        extracted_attn_weights.clear()
        # training initialization
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_results = []
        train_embeddings = {}
        all_train_labels = []
        all_train_probs = []
        all_train_preds = []
        
        if verbose:
            progress_training = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Train]') # start progress bar
        else:
            progress_training = train_loader
            
        for continuous_data, labels, patient_id in progress_training:
            # send data to GPU
            continuous_data = continuous_data.to(device, non_blocking=True).float() 
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            
            optimizer.zero_grad() # clear gradients
            outputs = model(continuous_data) # forward pass
            logits = outputs[1] # get pred logits from model output
            embeddings = outputs[0] # get embeddings from model output
            loss = criterion(logits, labels) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # update weights

            train_loss += loss.detach() * continuous_data.size(0) # accumulate loss
            probs = torch.sigmoid(logits).float() # convert logits to probabilities
            predicted = (probs > 0.5).float() # convert prediction to binary
            train_total += labels.size(0) # accumulate total samples
            train_correct += (predicted == labels).sum().detach() # accumulate correct predictions
            
            # store labels and predictions for metrics calculation
            all_train_labels.extend(labels.detach().cpu().numpy().flatten())
            all_train_probs.extend(probs.detach().cpu().numpy().flatten())
            all_train_preds.extend(predicted.detach().cpu().numpy().flatten())
            
            for pid, label, pred, prob in zip(patient_id, labels, predicted, probs):
                train_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
            
            for pid, label, emb in zip(patient_id, labels, embeddings):
                pid_item = pid.item()
                train_embeddings[pid_item] = {
                    'label' : int(label.item()),
                    'embedding' : emb.detach().cpu()#.numpy()
                }
            
            if verbose:
                progress_training.set_postfix(loss=train_loss / train_total, accuracy=100 * train_correct / train_total) # update progress bar
        
        # calculate auc and mcc
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        
        # validation
        with torch.no_grad(): # no gradient updates, only evaluation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_results = []
            val_embeddings = {}
            all_val_labels = []
            all_val_probs = []
            all_val_preds = []

            if verbose:
                progress_validation = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Val]')
            else:
                progress_validation = val_loader    
            
            for continuous_data, labels, patient_id in progress_validation:
                continuous_data = continuous_data.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float().view(-1, 1)
                
                outputs = model(continuous_data)
                logits = outputs[1]
                embeddings = outputs[0]
                loss = criterion(logits, labels)
                val_loss += loss.detach() * continuous_data.size(0)
                
                probs = torch.sigmoid(logits).float()
                predicted = (probs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().detach()

                all_val_labels.extend(labels.detach().cpu().numpy().flatten())
                all_val_probs.extend(probs.detach().cpu().numpy().flatten())
                all_val_preds.extend(predicted.detach().cpu().numpy().flatten())
                
                for pid, label, pred, prob in zip(patient_id, labels, predicted, probs):
                    val_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
                
                for pid, label, emb in zip(patient_id, labels, embeddings):
                    pid_item = pid.item()
                    val_embeddings[pid_item] = {
                        'label' : int(label.item()),
                        'embedding' : emb.detach().cpu()#.numpy()
                    }
                
                if verbose:
                    progress_validation.set_postfix(loss=val_loss / val_total, accuracy=100 * val_correct / val_total)
        
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        
        # calculate metrics for training and validation
        train_loss /= train_total
        train_accuracy = 100 * train_correct / train_total
        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total
        
        # for plotting
        train_losses.append(train_loss.item())
        train_accuracies.append(train_accuracy.item())
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy.item())
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_mccs.append(train_mcc)
        val_mccs.append(val_mcc)
        
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_results = train_results[:]
            best_val_results = val_results[:]
            best_train_embeddings = train_embeddings.copy()
            best_val_embeddings = val_embeddings.copy()
            
            # remove existing checkpoint files
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            
            train_result_df = pd.DataFrame(best_train_results, columns=['patient_id', 'label', 'prediction', 'probability'])
            val_result_df = pd.DataFrame(best_val_results, columns=['patient_id', 'label', 'prediction', 'probability'])
   
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'val_auc': val_auc,
                'train_auc': train_auc,
                'val_mcc': val_mcc,
                'train_mcc': train_mcc,
                'train_results': train_result_df,
                'val_results': val_result_df,
                'train_embeddings': best_train_embeddings,
                'val_embeddings': best_val_embeddings,
                'hyperparameters': {
                    'attention_dropout': model.attn_dropout,
                    'feedforward_dropout': model.ff_dropout,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'weight_decay': optimizer.param_groups[0]['weight_decay'],
                    'number_of_epochs': num_epochs,
                    'dimensions': best_config["dimensions"],
                    'depth': best_config["depth"],
                    'heads': best_config["heads"]
                }
            }, checkpoint_path)
        
        # epoch outputs
        if verbose: 
            print(f'\nEpoch {epoch}/{num_epochs-1}:')
            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, AUC: {train_auc:.4f}, MCC: {train_mcc:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}')
            print('-' * 60)
            
    if not verbose: 
        print(f'{num_epochs} epochs completed.')
            
        
    # plot
    epochs = range(num_epochs)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_aucs, label='Training AUC')
    plt.plot(epochs, val_aucs, label='Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Training and Validation AUC')
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_mccs, label='Training MCC')
    plt.plot(epochs, val_mccs, label='Validation MCC')
    plt.xlabel('Epochs')
    plt.ylabel('MCC')
    plt.legend()
    plt.title('Training and Validation MCC')
    plt.suptitle('Tabular Model Training Metrics')
    plt.tight_layout()
    plt.show()


"""
Best trial config: {'learning_rate': 4.8136972830665384e-05, 'weight_decay': 5.9369582068686936e-06, 'attention_dropout': 0.47614019114173, 'feedforward_dropout': 0.4604311614721739, 'batch_size': 32, 'dimensions': 64, 'depth': 6, 'heads': 4, 'epochs': 25, 'patience': 10}
Best trial final validation accuracy: 91.66666412353516
"""



# %%
def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    # tuned hyperparameters
    best_config = run_hyperparameter_tuning()
    
    final_df = preprocess_data()
    train_data, val_data, test_data = create_subset(final_df)
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)

    # load model
    model = TabularTransformer(
        attn_dropout=best_config["attention_dropout"],
        ff_dropout=best_config["feedforward_dropout"],
        dim=best_config["dimensions"],
        depth=best_config["depth"],
        heads=best_config["heads"]
    ).to(device)

    
    # class weights from training
    class_weights = compute_class_weights(train_data)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.AdamW(model.parameters(),
                            lr=best_config["learning_rate"],
                            weight_decay=best_config["weight_decay"])

    train_model(model, train_loader, val_loader, criterion, optimizer, best_config, num_epochs=number_of_epochs)

    torch.save(test_loader, os.path.join('tabular_checkpoints', 'test_loader.pth'))

    # not working
    #summary(model, input_size=(input_size,), device=device.type)
   
    # print class weights
    # 5.983739852905273
    print(f'Class Weights: {class_weights}')
   
    # cleanup to potentially run faster
    del train_loader, val_loader, test_loader, model, optimizer
    torch.cuda.empty_cache()
    
    
    features = [c for c in final_df.columns if c not in ['rlpnr', 'stressfx_Y']]
    # feature importances from last epoch
    attns = extracted_attn_weights[-1]  # shape: [num_layers, batch, heads, tokens, tokens]

    # average across batch, heads, layers
    cls_attn = attns[:, :, :, 0, 1:]  # [layers, batch, heads, features]
    feature_importance = cls_attn.mean(dim=(0, 1, 2))
    
    #print("Feature Importance: ", feature_importance)
    
    imp_pairs = list(zip(features, feature_importance.tolist()))
    imp_pairs.sort(key=lambda x: x[1], reverse=True)  # sort by importance
    
    for name, importance in imp_pairs:
        print(f"Feature: {name:20s}, Importance: {importance:.4f}")
    
if __name__ == "__main__":
    main()
# %%
