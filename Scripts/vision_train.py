# %%
import torch
import torch.nn as nn
import torch.optim as optim
from vision_preprocess import *
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import glob
import timm
from torchsummary import summary
from vision_tune import run_hyperparameter_tuning
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
num_epochs=100
patience_threshold=50


# define vision transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1, model_name='tiny_vit_5m_224.dist_in22k_ft_in1k'):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        #for param in self.model.head.parameters():
        #    param.requires_grad = True  # unfreeze the final classifier head
        
        
        # output single unit logit
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
         
        embeddings = self.model.forward_features(x) # get features from the transformer
        embeddings = embeddings.mean(dim=[2, 3]) # average pooling of height and width to match the expected input dimensions
        preds = self.model.forward_head(embeddings)  # output the logits for predictions
        
        # output embeddings and predictions
        return embeddings, preds

# training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, best_config, num_epochs=25, 
                patience_threshold=40, verbose = True, checkpoint_path = 'image_checkpoints/best_vision.pth'):
    best_val_loss = float('inf') # threshold for validation loss
    best_train_results = []
    best_val_results = []
    best_train_embeddings = {}
    best_val_embeddings = {}
    os.makedirs('image_checkpoints', exist_ok=True)
    
    # early stopping initialization
    early_stop_counter = 0
    
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
        # training
        model.train()
        # initialize metrics
        running_loss = 0.0
        correct = 0
        total = 0
        train_results = []
        train_embeddings = {}
        all_train_labels = []
        all_train_probs = []
        all_train_preds = []
        
        
        if verbose:
            progress_training = tqdm(train_loader, desc = f'Epoch {epoch}/{num_epochs-1} [Train]')
        else:
            progress_training = train_loader
        
        for patient_id, images, labels in progress_training:
            # send data to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            labels = labels.float().unsqueeze(1) # convert to float and add dimension
            
            optimizer.zero_grad() # clear gradients
            outputs = model(images) # forward pass
            logits = outputs[1] # predictions from forward pass
            embeddings = outputs[0] # embeddings from forward pass
            loss = criterion(logits, labels) # calculate loss
            loss.backward() # backpropagation
            optimizer.step() # update weights

            running_loss += loss.detach() * images.size(0) # accumulate loss
            probs = torch.sigmoid(logits).float() # convert logits to probabilities
            predicted = (probs > 0.5).long() # convert prediction to binary
            total += labels.size(0) # accumulate total samples
            correct += (predicted == labels).sum().detach() # accumulate correct predictions
            
            # store labels and predictions for metrics calculation
            all_train_labels.extend(labels.detach().cpu().numpy().flatten())
            all_train_probs.extend(probs.detach().cpu().numpy().flatten())
            all_train_preds.extend(predicted.detach().cpu().numpy().flatten())

            for pid, label, pred, prob in zip(patient_id, labels, predicted, probs):
                train_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
            
            for pid, label, emb in zip(patient_id, labels, embeddings):
                pid_item = pid.item()
                # unique identifier using batch index
                entry_id = f"{pid_item}_{len(train_embeddings.get(pid_item, []))}"
                if pid_item not in train_embeddings:
                    train_embeddings[pid_item] = []
                train_embeddings[pid_item].append({
                    'label': int(label.item()),
                    'embedding': emb.detach().cpu(),
                    'entry_id': entry_id
                })

            if verbose:
                # update progress bar
                progress_training.set_postfix(loss=running_loss / total, accuracy=100 * correct / total)
        # calculate auc and mcc
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        
        # validation
        with torch.no_grad():
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
                
            for patient_id, images, labels in progress_validation:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1)
                
                outputs = model(images)
                logits = outputs[1]
                embeddings = outputs[0]
                loss = criterion(logits, labels)
                val_loss += loss.detach() * images.size(0)
                
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
                    entry_id = f"{pid_item}_{len(val_embeddings.get(pid_item, []))}"
                    if pid_item not in val_embeddings:
                        val_embeddings[pid_item] = []
                    val_embeddings[pid_item].append({
                        'label': int(label.item()),
                        'embedding': emb.detach().cpu(),
                        'entry_id': entry_id
                    })
                
                if verbose:
                    progress_validation.set_postfix(loss=val_loss / val_total, accuracy=100 * val_correct / val_total)
        
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        
        # copmute metrics
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
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
            best_train_embeddings = train_embeddings
            best_val_embeddings = val_embeddings
            early_stop_counter = 0
            checkpoint_path
            
            # Remove existing checkpoint files
            for file in glob.glob(checkpoint_path):
                os.remove(file)
            
            train_result_df = pd.DataFrame(best_train_results, columns=['patient_id', 'label', 'prediction', 'probability'])
            val_result_df = pd.DataFrame(best_val_results, columns=['patient_id', 'label', 'prediction', 'probability'])
   
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
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
                    'pretrained_lr': best_config['pretrained_lr'],
                    'finetuning_lr': best_config['finetuning_lr'],
                    'weight_decay': best_config['weight_decay'],
                    'batch_size': best_config['batch_size'],
                    'num_epochs': num_epochs,
                    'patience': patience_threshold
                }
            }, checkpoint_path)
        else:
            early_stop_counter +=1 
            
        if early_stop_counter >= patience_threshold:
            if verbose:
                print('-' * 60)
                print('\n')
                print(f"Early stopping after {epoch+1} epochs")
                print('-' * 60)
            else:
                print(f"Early stopping after {epoch+1} epochs")
            break
        
        if verbose:
            # epoch outputs
            print(f'\nEpoch {epoch}/{num_epochs-1} Summary:')
            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, AUC: {train_auc:.4f}, MCC: {train_mcc:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}')
            print('-' * 60)
            
    if not verbose:
        print(f"{num_epochs} epochs completed")
        
    # plot
    epochs = range(len(train_losses))
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
    plt.suptitle('Vision Model Training Metrics')
    plt.tight_layout()
    plt.show()

# Main function
def main():
    start_time = time.time()
    if os.path.exists('test_loader.pth'):
        os.remove('test_loader.pth')
    if os.path.exists('class_weights.pth'):
        os.remove('class_weights.pth')
    
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    images_start_time = time.time()
    transform = previous_augmentations()
    images = ImgData(source_path, transform=transform)
    images_end_time = time.time()
    
    train_data, val_data, test_data = create_subset(images)
    class_weights = images.class_weights
    
    init_batch_size = 16
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=init_batch_size
    )

    tuning_start_time = time.time()
    best_config = run_hyperparameter_tuning(
        train_loader=train_loader, 
        val_loader=val_loader, 
        class_weights=class_weights
    )
    tuning_end_time = time.time()
    
    if best_config["batch_size"] != init_batch_size:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data, val_data, test_data, 
            batch_size=best_config["batch_size"]
        )    
    
    model = VisionTransformer(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # create separate parameters for pretrained and finetuning
    pretrained_params = [param for name, param in model.named_parameters() if 'head' not in name]
    finetuning_params = [param for name, param in model.named_parameters() if 'head' in name]
    
    # use the different learning rates
    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': best_config["pretrained_lr"]},
        {'params': finetuning_params, 'lr': best_config["finetuning_lr"]}
    ], weight_decay=best_config["weight_decay"])
    
    #optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    training_start_time = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                best_config, num_epochs=num_epochs, patience_threshold=patience_threshold)
    training_end_time = time.time()
    
    # save testdata and class weights for exporting
    torch.save(test_loader, os.path.join('image_checkpoints','image_test_loader.pth'))
    torch.save(class_weights, os.path.join('image_checkpoints','class_weights.pth'))

    # 5.7484, 6.3248
    print(f"Class weights: {class_weights}")
    print("\n")

    # model summary
    summary(model, input_size=(3, 224, 224))

    # cleanup to potentially run faster
    del train_loader, val_loader, test_loader, model, optimizer
    torch.cuda.empty_cache()
    
    for p in mp.active_children():
        p.join()
    
    end_time = time.time()
    
    # Print accumulated run time
    total_time = end_time - start_time
    process_time = images_end_time - images_start_time
    tuning_time = tuning_end_time - tuning_start_time
    training_time = training_end_time - training_start_time
    
    print(f"Total run time: {total_time:.2f} seconds")
    print(f"Processing time: {process_time:.2f} seconds")
    print(f"Hyperparameter tuning time: {tuning_time:.2f} seconds")
    print(f"Training time: {training_time:.2f} seconds")
if __name__ == "__main__":
    main()
# %%