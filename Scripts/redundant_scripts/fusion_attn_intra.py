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
from extract_embeddings import pair_test_emb, pair_train_emb, pair_val_emb
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# better feature learning within each modality
# apply self-attention independelty on the final layer of embeddings from both unimodal
# this allows for better feature learning within each modality
# the refined representations are concatenated
# %%
pair_train_emb[0]


# %%
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)

class IntraModalitySelfAttentionFusion(nn.Module):
    def __init__(self, vision_dim=768, tabular_dim=32, fused_dim=128, num_classes=1):
        super(IntraModalitySelfAttentionFusion, self).__init__()

        # unify dimensions
        self.vision_proj = ProjectionLayer(vision_dim, fused_dim)
        self.tabular_proj = ProjectionLayer(tabular_dim, fused_dim)

        # self-attention layers
        self.self_attention_tab = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4)
        self.self_attention_vision = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4)

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

        # self-attention to each modality separately
        # modify so attention weights are used for visualization
        attn_tab, weights_tab = self.self_attention_tab(tabular_embeddings, tabular_embeddings, tabular_embeddings)
        attn_vision, weights_vision = self.self_attention_vision(vision_embeddings, vision_embeddings, vision_embeddings)

        # concatenate new representations and classify
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

def train_intra_fusion_model(model, train_data, val_data, num_epochs=200, batch_size=32, learning_rate=1e-4):
    os.makedirs('intra_fusion_checkpoints', exist_ok=True)
    
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
    
    # loss and optimizer
    pos_weight = compute_class_weights(train_labels)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # metrics
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
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Train]')
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
            
            # append results with pat IDs
            for pid, label, pred, prob in zip(patient_id, labels, predicted, outputs):
                train_results.append((pid.cpu().item(), 
                                      int(label.cpu().item()), 
                                      int(pred.cpu().item()), 
                                      float(prob.cpu().item())))
            
            progress_bar.set_postfix(loss=train_loss/train_total, 
                                    accuracy=100*train_correct/train_total)
        
        # calculate auc and mcc
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        
        # validation loop
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
                
                progress_bar.set_postfix(loss=val_loss/val_total, 
                                        accuracy=100*val_correct/val_total)
        
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        
        # epoch metrics
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
            
            train_result_df = pd.DataFrame(best_train_results, 
                                          columns=['patient_id', 'label', 'prediction', 'probability'])
            val_result_df = pd.DataFrame(best_val_results, 
                                        columns=['patient_id', 'label', 'prediction', 'probability'])
            
            # checkpoint
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
            }, 'intra_fusion_checkpoints/best_fusion.pth')
            
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

def test_intra_fusion_model(model, test_data):
    # set checkpoint path
    checkpoint_path = 'intra_fusion_checkpoints/best_fusion.pth'
    
    # load the best model
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # extract test data
    test_tab_emb = torch.stack([item.tabular for item in test_data]).to(device)
    test_img_emb = torch.stack([item.vision for item in test_data]).to(device)
    test_patient_ids = torch.tensor([item.pid for item in test_data]).long()
    
    # extract labels for test
    test_labels_dict = {item.pid: item.label for item in test_data}
    test_labels = torch.tensor([test_labels_dict[pid.item()] for pid in test_patient_ids]).float().view(-1, 1).to(device)
    
    # initialize metrics
    test_loss = 0.0
    test_correct = 0
    test_total = len(test_labels)
    test_results = []
    test_predictions = []
    
    # loss function
    criterion = nn.BCELoss()
    
    # evaluate model
    with torch.no_grad():
        # forward pass
        outputs = model(test_tab_emb, test_img_emb)
        loss = criterion(outputs, test_labels)
        predicted = (outputs > 0.5).float()
        
        # calculate metrics
        test_loss += loss.item() * test_total
        test_correct += (predicted == test_labels).sum().item()
        
        for pid, label, pred, prob in zip(test_patient_ids, test_labels, predicted, outputs):
            test_results.append((pid.item(), 
                                 int(label.cpu().item()), 
                                 int(pred.cpu().item()), 
                                 float(prob.cpu().item())))
            test_predictions.append(int(pred.cpu().item()))
    
    # calculate final metrics + precision, recall, and F1-score
    test_loss /= test_total
    test_accuracy = 100 * test_correct / test_total
    test_labels_np = test_labels.cpu().numpy()
    precision, recall, f1, _ = score(test_labels_np, test_predictions, average='binary', zero_division=0)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}, Test Recall (Sensitivity): {recall:.4f}, Test F1-Score: {f1:.4f}')
    
    # save results
    test_result_df = pd.DataFrame(test_results, 
                                 columns=['patient_id', 'label', 'prediction', 'probability'])
    test_result_df.to_csv('intra_fusion_checkpoints/test_results.csv', index=False)
    
    # confusion matrix
    cm = confusion_matrix(test_labels_np, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Intra-Attention Fusion Confusion Matrix')
    plt.savefig('intra_fusion_checkpoints/confusion_matrix.png')
    plt.show()
    
    return test_result_df

def aggregate_patient_results(test_results_df):
    # aggregate probabilities by patient
    patient_aggregated = test_results_df.groupby('patient_id').agg({
        'label': 'first',  # label is identical for each patient
        'probability': 'mean'  # average probs
    }).reset_index()
    
    # new predictions based on aggregation
    patient_aggregated['prediction'] = (patient_aggregated['probability'] > 0.5).astype(int)
    
    # accuracy, arecision, recall, F1
    accuracy = (patient_aggregated['prediction'] == patient_aggregated['label']).mean() * 100
    
    precision, recall, f1, _ = score(
        patient_aggregated['label'], 
        patient_aggregated['prediction'], 
        average='binary', 
        zero_division=0
    )
    
    print("\nPatient-level Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # confusion matrix for patient-level predictions
    cm = confusion_matrix(patient_aggregated['label'], patient_aggregated['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Intra-Attention Fusion Patient-Level Confusion Matrix')
    plt.savefig('intra_fusion_checkpoints/patient_level_confusion_matrix.png')
    plt.show()
    
    return patient_aggregated

if __name__ == "__main__":
    # instantiate the model
    intra_fusion_model = IntraModalitySelfAttentionFusion(
        vision_dim=576, 
        tabular_dim=64, 
        fused_dim=128, 
        num_classes=1
    ).to(device)
    
    # train the model
    intra_fusion_model = train_intra_fusion_model(
        intra_fusion_model,
        pair_train_emb,
        pair_val_emb,
        num_epochs=200,
        batch_size=32,
        learning_rate=1e-6
    )
    
    # test the model
    test_results_df = test_intra_fusion_model(intra_fusion_model, pair_test_emb)
    
    # aggregate analysis
    patient_level_results = aggregate_patient_results(test_results_df)
# %%