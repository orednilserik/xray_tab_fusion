# %%
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from vision_train import VisionTransformer
from tabular_train import TabularTransformer
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support as score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from extract_embeddings import extract_embeddings
from sklearn.metrics import roc_auc_score, matthews_corrcoef

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

class InterModalityCrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim=320, tabular_dim=64, fused_dim=128, num_classes=1):
        super(InterModalityCrossAttentionFusion, self).__init__()

        # unify dimensions
        self.vision_proj = ProjectionLayer(vision_dim, fused_dim)
        self.tabular_proj = ProjectionLayer(tabular_dim, fused_dim)

        # cross-attention layers
        self.cross_attention_tab = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4)
        self.cross_attention_vision = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4)

        # post-fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, tabular_embeddings, vision_embeddings):
        # project embeddings same size
        tabular_embeddings = self.tabular_proj(tabular_embeddings)
        vision_embeddings = self.vision_proj(vision_embeddings)

        # cross-attention between each modality
        # modify so attention weights are used for visualization
        attn_tab, weights_tab = self.cross_attention_tab(tabular_embeddings, vision_embeddings, vision_embeddings)
        attn_vision, weights_vision = self.cross_attention_vision(vision_embeddings, tabular_embeddings, tabular_embeddings)

        # Concatenate and classify
        fused_features = torch.cat((attn_tab, attn_vision), dim=1)
        fused_preds = self.fusion(fused_features)

        return fused_preds

def compute_class_weights(labels_tensor):
    # ¨make sure tensor is on CPU before converting to numpy
    if torch.is_tensor(labels_tensor) and labels_tensor.is_cuda:
        labels_tensor = labels_tensor.cpu()
    
    labels = np.array(labels_tensor).flatten()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels)
    pos_weight = torch.tensor(class_weights.max() / class_weights.min(), dtype=torch.float32, device=device)
    return pos_weight

def train_inter_fusion_model(model, train_data, val_data, num_epochs=200, batch_size=32, 
                             learning_rate=1e-4, weight_decay=1e-3, fold_no=None, verbose=True):
    
    checkpoint_dir = 'inter_fusion_checkpoints'
    if fold_no is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f'fold_{fold_no}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # extract labels and patient IDs safely
    train_labels_dict = {item.pid: item.label for item in train_data}
    val_labels_dict = {item.pid: item.label for item in val_data}
    
    # move to CPU first to avoid any GPU->NumPy conversions
    train_tab_emb = torch.stack([item.tabular.cpu() if torch.is_tensor(item.tabular) and item.tabular.is_cuda else item.tabular for item in train_data])
    train_img_emb = torch.stack([item.vision.cpu() if torch.is_tensor(item.vision) and item.vision.is_cuda else item.vision for item in train_data])
    train_patient_ids = torch.tensor([item.pid for item in train_data]).long()
    
    val_tab_emb = torch.stack([item.tabular.cpu() if torch.is_tensor(item.tabular) and item.tabular.is_cuda else item.tabular for item in val_data])
    val_img_emb = torch.stack([item.vision.cpu() if torch.is_tensor(item.vision) and item.vision.is_cuda else item.vision for item in val_data])
    val_patient_ids = torch.tensor([item.pid for item in val_data]).long()
    
    # extract labels
    train_labels = torch.tensor([train_labels_dict[pid.item()] for pid in train_patient_ids]).float().view(-1, 1)
    val_labels = torch.tensor([val_labels_dict[pid.item()] for pid in val_patient_ids]).float().view(-1, 1)
    
    # create CPU datasets and move them to device in batches
    train_dataset = TensorDataset(train_tab_emb, train_img_emb, train_labels, train_patient_ids)
    val_dataset = TensorDataset(val_tab_emb, val_img_emb, val_labels, val_patient_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # define loss and optimizer
    pos_weight = compute_class_weights(train_labels)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # tracking metrics
    best_val_loss = float('inf')
    best_train_results = []
    best_val_results = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_aucs = []
    val_aucs = []
    train_mccs = []
    val_mccs = []
    
    # training loop
    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_results = []
        all_train_labels = []
        all_train_probs = []
        all_train_preds = []
        
        if verbose:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Train]')
        else:
            progress_bar = train_loader
            
        for tab_emb, img_emb, labels, patient_id in progress_bar:
            tab_emb = tab_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)
            
            # forward pass
            optimizer.zero_grad()
            outputs = model(tab_emb, img_emb)
            loss = criterion(outputs, labels)
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            # calculate metrics
            train_loss += loss.item() * labels.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # store labels and predictions for metrics calculation
            all_train_labels.extend(labels.detach().cpu().numpy().flatten())
            all_train_probs.extend(outputs.detach().cpu().numpy().flatten())
            all_train_preds.extend(predicted.detach().cpu().numpy().flatten())
            
            for pid, label, pred, prob in zip(patient_id, labels, predicted, outputs):
                train_results.append((pid.cpu().item(), 
                                      int(label.cpu().item()), 
                                      int(pred.cpu().item()), 
                                      float(prob.cpu().item())))
            
            if verbose:
                progress_bar.set_postfix(loss=train_loss/train_total, 
                                        accuracy=100*train_correct/train_total)
        
         # calculate auc and mcc
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        
        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_results = []
        all_val_labels = []
        all_val_probs = []
        all_val_preds = []
        
        with torch.no_grad():
            if verbose:
                progress_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Val]')
            else:
                progress_bar = val_loader
            for tab_emb, img_emb, labels, patient_id in progress_bar:
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
                all_val_preds.extend(predicted.detach().cpu().numpy().flatten())
                
                for pid, label, pred, prob in zip(patient_id, labels, predicted, outputs):
                    val_results.append((pid.cpu().item(), 
                                        int(label.cpu().item()), 
                                        int(pred.cpu().item()), 
                                        float(prob.cpu().item())))
                
                if verbose:
                    progress_bar.set_postfix(loss=val_loss/val_total, 
                                            accuracy=100*val_correct/val_total)
                
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        
        # calculate epoch metrics
        train_loss /= train_total
        train_accuracy = 100 * train_correct / train_total
        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total
        
        # save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_mccs.append(train_mcc)
        val_mccs.append(val_mcc)
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_results = train_results[:]
            best_val_results = val_results[:]
            
            # convert to dataframes for easy saving
            train_result_df = pd.DataFrame(best_train_results, 
                                          columns=['patient_id', 'label', 'prediction', 'probability'])
            val_result_df = pd.DataFrame(best_val_results, 
                                        columns=['patient_id', 'label', 'prediction', 'probability'])
            
            # save checkpoint
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
                'val_results': val_result_df
            }, os.path.join(checkpoint_dir, 'best_fusion.pth'))
            
        # print epoch results
        if verbose:
            print(f'\nEpoch {epoch}/{num_epochs-1}:')
            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, AUC: {train_auc:.4f}, MCC: {train_mcc:.4f}')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}')
            print('-' * 60)
    
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_curves.png'))
    plt.show()
    
    return model

def enable_dropout(model):
    # enable dropout layers for MC dropout
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # set dropout layers to training mode duringi nference
            

def test_inter_fusion_mc(model, test_data, num_samples=100, fold_no=None):
    checkpoint_dir = 'inter_fusion_checkpoints'
    if fold_no is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f'fold_{fold_no}')
    checkpoint_path = os.path.join(checkpoint_dir, 'best_fusion.pth')
    
    # Load the best model
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # enable dropout for uncertainty estimation
    enable_dropout(model)
    
    # Extract test data
    test_tab_emb = torch.stack([item.tabular for item in test_data]).to(device)
    test_img_emb = torch.stack([item.vision for item in test_data]).to(device)
    test_patient_ids = torch.tensor([item.pid for item in test_data]).long()
    
    # Extract labels for test
    test_labels_dict = {item.pid: item.label for item in test_data}
    test_labels = torch.tensor([test_labels_dict[pid.item()] for pid in test_patient_ids]).float().view(-1, 1).to(device)
    
    # initialize metrics
    test_total = len(test_labels)
    test_mc_predictions = []
    
    # perform multiple stochastic forward passes
    print(f"Performing {num_samples} stochastic forward passes with MC Dropout")
    progress_bar = tqdm(range(num_samples))
    
    for _ in progress_bar:
        # forward pass with dropout enabled
        with torch.no_grad():
            outputs = model(test_tab_emb, test_img_emb)
            test_mc_predictions.append(outputs.cpu())
            
    # stacking all MC samples
    test_mc_predictions = torch.stack(test_mc_predictions)
    
    # calculate mean and std dev for samples
    mean_preds = test_mc_predictions.mean(dim=0)
    std_preds = test_mc_predictions.std(dim=0)
    
    # make binary preds based on mean
    predicted = (mean_preds > 0.5).float()
    
    # numpy for sklearn metrics
    test_labels_np = test_labels.cpu().numpy()
    mean_preds_np = mean_preds.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    
    test_auc = roc_auc_score(test_labels_np, mean_preds_np)
    test_mcc = matthews_corrcoef(test_labels_np.flatten(), predicted_np.flatten())
    
    # storing results
    test_results= []
    for pid, label, pred, prob, unc in zip(test_patient_ids, test_labels, predicted,mean_preds, std_preds):
        test_results.append((pid.item(),
                             int(label.cpu().item()), 
                             int(pred.cpu().item()), 
                             float(prob.cpu().item()),
                             float(unc.cpu().item())))
    
    # calculate metrics
    test_correct = (predicted == test_labels.cpu()).sum().item()
    test_accuracy = 100 * test_correct / test_total
    test_preds = [int(pred.cpu().item()) for pred in predicted]
    precision, recall, f1, _ = score(test_labels_np, test_preds, average='binary', zero_division=0)
    
    print(f'MC Dropout Test Accuracy: {test_accuracy:.2f}%')
    print(f'MC Dropout Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')
    print(f'MC Dropout Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1-Score: {f1:.4f}')
    
    # Save results to DataFrame
    test_result_df = pd.DataFrame(test_results, 
                                 columns=['patient_id', 'label', 'prediction', 'probability', 'uncertainty'])
    test_result_df.to_csv(os.path.join(checkpoint_dir, 'test_results.csv'), index=False)
    
    # visualization: uncertainty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_result_df['uncertainty'], bins=20, alpha=0.7)
    plt.xlabel('Prediction Uncertainty (std)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Uncertainty using MC Dropout')
    plt.savefig(os.path.join(checkpoint_dir, 'uncertainty_distribution.png'))
    plt.show()
    
    # visualization: uncertainty vs correctness
    correct_preds = (test_result_df['prediction'] == test_result_df['label']).astype(int)
    plt.figure(figsize=(10, 6))
    plt.scatter(test_result_df['probability'], test_result_df['uncertainty'], 
               c=correct_preds, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Prediction Correctness (1=Correct, 0=Incorrect)')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Prediction Uncertainty (std)')
    plt.title('Prediction Probability vs Uncertainty with MC Dropout')
    plt.savefig(os.path.join(checkpoint_dir, 'uncertainty_vs_probability.png'))
    plt.show()
    
    # confusion matrix
    cm = confusion_matrix(test_labels_np, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Inter-Attention Fusion Confusion Matrix (MC Dropout)')
    plt.savefig(os.path.join(checkpoint_dir, 'confusion_matrix_mc.png'))
    plt.show()
    
    return test_result_df

def aggregate_patient_results_mc(test_results_df, inv_weight = True,fold_no=None):
    
    results = test_results_df.copy()
    
    checkpoint_dir = 'inter_fusion_checkpoints'
    if fold_no is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, f'fold_{fold_no}')
    
    # calculate weight by inverse of uncertainty
    if inv_weight:
        epsilon = 1e-6 # avoid division by 0
        results['weight'] = 1.0 / (results['uncertainty'] + epsilon)
    else:
        results['weight'] = 1.0
    
    patient_results = (
        results.groupby('patient_id')
        .apply(lambda x: pd.Series({
            'label': x['label'].iloc[0],  # all labels are the same for each patient
            'weighted_probability': (x['probability'] * x['weight']).sum() / x['weight'].sum(),
            'mean_uncertainty': x['uncertainty'].mean(),
            'num_samples': len(x)
        }))
        .reset_index()
    )
    
    # new predictions based on aggregated probabilities
    patient_results['prediction'] = (patient_results['weighted_probability'] > 0.5).astype(int)
    
    # compute metrics
    accuracy = (patient_results['prediction'] == patient_results['label']).mean() * 100
    
    precision, recall, f1, _ = score(
        patient_results['label'], 
        patient_results['prediction'], 
        average='binary', 
        zero_division=0
    )
    
    patient_auc = roc_auc_score(patient_results['label'], patient_results['weighted_probability'])
    patient_mcc = matthews_corrcoef(patient_results['label'], patient_results['prediction'])
    
    print("\nPatient-level Metrics with MC Dropout:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC: {patient_auc:.4f}, MCC: {patient_mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save results
    patient_results.to_csv(os.path.join(checkpoint_dir, 'patient_results_mc_weighted.csv'), index=False)
    
    # confusion matrix for patient-level predictions
    cm = confusion_matrix(patient_results['label'], patient_results['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Inter-Attention Fusion Patient-Level Confusion Matrix (MC Dropout)')
    plt.savefig(os.path.join(checkpoint_dir, 'patient_level_confusion_matrix_mc.png'))
    plt.show()
    
    # Uncertainty analysis at patient-level
    plt.figure(figsize=(10, 6))
    plt.scatter(patient_results['weighted_probability'], patient_results['mean_uncertainty'], 
               c=(patient_results['prediction'] == patient_results['label']).astype(int), 
               cmap='coolwarm', alpha=0.7, s=100)
    plt.colorbar(label='Prediction Correctness (1=Correct, 0=Incorrect)')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Prediction Uncertainty (mean std)')
    plt.title('Patient-Level Prediction Probability vs Uncertainty')
    plt.savefig(os.path.join(checkpoint_dir, 'patient_level_uncertainty.png'))
    plt.show()
    
    return patient_results


if __name__ == "__main__":
    for fold_no in range(1,6):
        # instantiate the model
        inter_fusion_model = InterModalityCrossAttentionFusion().to(device)
        
        pair_train_emb, pair_val_emb, pair_test_emb = extract_embeddings(f'fold_{fold_no}')
        
        # train the model
        fusion_trained = train_inter_fusion_model(
            inter_fusion_model,
            pair_train_emb,
            pair_val_emb,
            num_epochs=200,
            batch_size=32,
            learning_rate=1e-6,
            fold_no=fold_no
        )
    
        # MC Dropout evaluation for uncertainty estimation
        test_results_mc_df = test_inter_fusion_mc(fusion_trained, pair_test_emb, num_samples=100,fold_no=fold_no)
        patient_level_results_mc = aggregate_patient_results_mc(test_results_mc_df,fold_no=fold_no)
    
    
# %%