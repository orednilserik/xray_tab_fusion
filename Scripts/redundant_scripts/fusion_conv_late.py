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
from tabular_preprocess import compute_class_weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support as score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#from extract_probabilities import merged_train_df, merged_val_df, merged_test_df
from extract_embeddings import pair_test_emb, pair_train_emb, pair_val_emb
from sklearn.metrics import roc_auc_score, matthews_corrcoef
# %%
def compute_class_weights(labels_tensor):
    labels = np.array(labels_tensor).flatten()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels) # compute class weights
    # compute positive ratio for bcewithlogitsloss
    pos_weight = torch.tensor(class_weights.max() / class_weights.min(), dtype=torch.float32, device=device)
    return pos_weight



# LATE FUSION ON PROBABILITIES
class LateFusion(nn.Module):
    def __init__(self, num_classes=1):
        super(LateFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, tabular_prob, vision_prob):
        # concatenate the predictions
        preds = torch.cat((tabular_prob, vision_prob), dim=1)
        
        # pass the concatenated embeddings through the classifier
        fused_preds = self.fusion(preds)
        
        return fused_preds
    
def train_fusion_model(model, train_data, val_data, num_epochs=25, batch_size=32, learning_rate=1e-3):
    os.makedirs('late_fusion_checkpoints', exist_ok=True)

    """
    This is previous code when using probabilities
    # extract img and tab predictions + label from datasets for both training and val
    X_train_tab = torch.tensor(train_df['probability_tab'].values).float().view(-1, 1)
    X_train_img = torch.tensor(train_df['probability_img'].values).float().view(-1, 1)
    y_train = torch.tensor(train_df['label'].values).float().view(-1, 1)
    
    X_val_tab = torch.tensor(val_df['probability_tab'].values).float().view(-1, 1)
    X_val_img = torch.tensor(val_df['probability_img'].values).float().view(-1, 1)
    y_val = torch.tensor(val_df['label'].values).float().view(-1, 1)
    
    train_patient_id = torch.tensor(train_df['patient_id'].values).long()
    val_patient_id = torch.tensor(val_df['patient_id'].values).long()
    """
        # extract labels and patient IDs safely
    train_labels_dict = {item.pid: item.label for item in train_data}
    val_labels_dict = {item.pid: item.label for item in val_data}
    
    # move to CPU first
    train_tab_emb = torch.stack([item.tabular.cpu() if torch.is_tensor(item.tabular) and item.tabular.is_cuda else item.tabular for item in train_data])
    train_img_emb = torch.stack([item.vision.cpu() if torch.is_tensor(item.vision) and item.vision.is_cuda else item.vision for item in train_data])
    train_patient_ids = torch.tensor([item.pid for item in train_data]).long()
    
    val_tab_emb = torch.stack([item.tabular.cpu() if torch.is_tensor(item.tabular) and item.tabular.is_cuda else item.tabular for item in val_data])
    val_img_emb = torch.stack([item.vision.cpu() if torch.is_tensor(item.vision) and item.vision.is_cuda else item.vision for item in val_data])
    val_patient_ids = torch.tensor([item.pid for item in val_data]).long()
    
   # extract labels
    train_labels = torch.tensor([train_labels_dict[pid.item()] for pid in train_patient_ids]).float().view(-1, 1)
    val_labels = torch.tensor([val_labels_dict[pid.item()] for pid in val_patient_ids]).float().view(-1, 1)
    
    # create CPU datasets and then move to device in batches
    train_dataset = TensorDataset(train_tab_emb, train_img_emb, train_labels, train_patient_ids)
    val_dataset = TensorDataset(val_tab_emb, val_img_emb, val_labels, val_patient_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # define loss and optimizer
    pos_weight = compute_class_weights(train_labels)

    criterion = nn.BCELoss(weight=pos_weight) # sigmoid activation implemented in last layer
    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-4, weight_decay=1e-3)
    
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
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_results = []
        all_train_labels = []
        all_train_probs = []
        all_train_preds = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Train]')
        for tab_probs, img_probs, labels, patient_id in progress_bar:
            tab_probs = tab_probs.to(device)
            img_probs = img_probs.to(device)
            labels = labels.to(device)
            
            # forward pass
            optimizer.zero_grad()
            outputs = model(tab_probs, img_probs)
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
                train_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
            
            
            progress_bar.set_postfix(loss=train_loss/train_total, 
                                     accuracy=100*train_correct/train_total)
            
        # calculate auc and mcc
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
            
    # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_results = []
        all_val_labels = []
        all_val_probs = []
        all_val_preds = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Val]')
            for tab_probs, img_probs, labels, patient_id in progress_bar:
                tab_probs = tab_probs.to(device)
                img_probs = img_probs.to(device)
                labels = labels.to(device)
                
                outputs = model(tab_probs, img_probs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_labels.extend(labels.detach().cpu().numpy().flatten())
                all_val_probs.extend(outputs.detach().cpu().numpy().flatten())
                all_val_preds.extend(predicted.detach().cpu().numpy().flatten())
                
                for pid, label, pred, prob in zip(patient_id, labels, predicted, outputs):
                    val_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
                
                progress_bar.set_postfix(loss=val_loss/val_total, 
                                         accuracy=100*val_correct/val_total)
        
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        
        # Calculate epoch metrics
        train_loss /= train_total
        train_accuracy = 100 * train_correct / train_total
        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total
        
        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_mccs.append(train_mcc)
        val_mccs.append(val_mcc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_results = train_results[:]
            best_val_results = val_results[:]
            
            # Convert to dataframes for easy saving
            train_result_df = pd.DataFrame(best_train_results, 
                                          columns=['patient_id', 'label', 'prediction', 'probability'])
            val_result_df = pd.DataFrame(best_val_results, 
                                        columns=['patient_id', 'label', 'prediction', 'probability'])
            
            # Save checkpoint
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
            }, 'late_fusion_checkpoints/best_fusion.pth')
            
        # Print epoch results
        print(f'\nEpoch {epoch}/{num_epochs-1}:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, AUC: {train_auc:.4f}, MCC: {train_mcc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}')
        print('-' * 60)
        
        

 # plots
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
    plt.savefig('intra_fusion_checkpoints/training_curves.png')
    plt.show()
    
    return model

def test_fusion_model(model, test_df):
    # Set checkpoint path
    checkpoint_path = 'late_fusion_checkpoints/best_fusion.pth'
    
    # Load the best model
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract data from test dataframe
    X_test_tab = torch.tensor(test_df['probability_tab'].values).float().view(-1, 1).to(device)
    X_test_img = torch.tensor(test_df['probability_img'].values).float().view(-1, 1).to(device)
    y_test = torch.tensor(test_df['label'].values).float().view(-1, 1).to(device)
    patient_ids_test = torch.tensor(test_df['patient_id'].values).long()
    
    # Initialize metrics
    test_loss = 0.0
    test_correct = 0
    test_total = len(y_test)
    test_results = []
    test_labels = []
    test_predictions = []
    
    criterion = nn.BCELoss()
    
    # Evaluate model
    with torch.no_grad():
        # Forward pass
        outputs = model(X_test_tab, X_test_img)
        loss = criterion(outputs, y_test)
        predicted = (outputs > 0.5).float()
        
        # Calculate metrics
        test_loss += loss.item() * test_total
        test_correct += (predicted == y_test).sum().item()
        test_labels.extend(y_test.cpu().numpy())
        test_predictions.extend(predicted.cpu().numpy())
        
        # Store results with patient IDs
        for pid, label, pred, prob in zip(patient_ids_test, y_test, predicted, outputs):
            test_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
    
    # Calculate final metrics
    test_loss /= test_total
    test_accuracy = 100 * test_correct / test_total
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = score(test_labels, test_predictions, average='binary', zero_division=0)
    
    # Print metrics
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}, Test Recall (Sensitivity): {recall:.4f}, Test F1-Score: {f1:.4f}')
    
    # Save results to DataFrame
    test_result_df = pd.DataFrame(test_results, 
                                 columns=['patient_id', 'label', 'prediction', 'probability'])
    test_result_df.to_csv('late_fusion_checkpoints/test_results.csv', index=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    disp.plot(cmap='Blues')
    plt.title('Late Fusion Confusion Matrix')
    plt.savefig('late_fusion_checkpoints/confusion_matrix.png')
    plt.show()
    
    return test_result_df

if __name__ == "__main__":
    # Create and train fusion model
    fusion_model = LateFusion().to(device)
    
    # Train the model
    fusion_trained = train_fusion_model(
        fusion_model, 
        pair_train_emb,
        pair_val_emb,
        num_epochs=200, 
        batch_size=32, 
        learning_rate=1e-6
    )
    
    
    test_results_df = test_fusion_model(fusion_trained, pair_test_emb)

# %%

############### AGGREGATIONS ON LATE FUSION

fusion_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/best_fusion.pth'

# read data and load dfs
fusion_checkpoint = torch.load(fusion_path, weights_only=False)
fusion_train_df = fusion_checkpoint['train_results']
fusion_val_df = fusion_checkpoint['val_results']
fusion_test_df = pd.read_csv('/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/test_results.csv')
fusion_test_df = fusion_test_df.drop(columns=['Unnamed: 0'])
print(fusion_train_df.head())
print(fusion_val_df.head())
print(fusion_test_df.head())
print(fusion_train_df.shape)
print(fusion_val_df.shape)
print(fusion_test_df.shape)
# %%

# aggregate probabilities
patient_aggregated = fusion_test_df.groupby('patient_id').agg({
    'label': 'first',  # Label is the same for all rows of a patient
    'probability': 'mean'  # Average the probabilities
}).reset_index()

# recalculate predictions
patient_aggregated['prediction'] = (patient_aggregated['probability'] > 0.5).astype(int)

# calculate accuracy
accuracy = (patient_aggregated['prediction'] == patient_aggregated['label']).mean() * 100

# precision, recall, F1
precision, recall, f1, _ = score(
    patient_aggregated['label'], 
    patient_aggregated['prediction'], 
    average='binary', 
    zero_division=0
)

# Display metrics
print("\nPatient-level Metrics:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot confusion matrix for patient-level predictions
cm = confusion_matrix(patient_aggregated['label'], patient_aggregated['prediction'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues')
plt.title('Late Fusion Patient-Level Confusion Matrix')
plt.savefig('late_fusion_checkpoints/patient_level_confusion_matrix.png')
plt.show()
# %%
