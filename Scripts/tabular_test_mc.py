# %%
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support as score
from tab_transformer_pytorch import FTTransformer
from tabular_preprocess import *
import random
from tqdm import tqdm
import os
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = 'best_tabular.pth'

training_checkpoints = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'

input_size = 45 # amount of variables put into model
num_classes = 1 # binary classification

# define tabular transformer model with explicit dropout
class TabularTransformer(nn.Module):
    def __init__(self, attn_dropout=0.2, ff_dropout=0.2, dim=32, depth=6, heads=8, dropout_rate=0.5):
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
        
        # explicit dropout for MC Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # separate classifier head for binary prediction
        self.classifier = nn.Sequential(
            self.dropout,
            nn.Linear(dim, num_classes)
        )
            
    def forward(self, x):
        categorical_data = x[:, 0].unsqueeze(1).long() # categorical data
        continuous_data = x[:, 1:] # continuous data
        
        embeddings = self.model(categorical_data, continuous_data) # get features from the transformer
        preds = self.classifier(embeddings) # output the logits for predictions
                
        return embeddings, preds

# class weights
def compute_class_weights(dataset):
    labels = np.array([label for _, label,_ in dataset]) # extract labels
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels) # compute class weights
    # compute ratio for bcewithlogitsloss
    pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32, device=device)
    return pos_weight

def enable_dropout(model):
    # enable dropout layers for MC dropout inference
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # set dropout layers to training mode during inference

def test_model(model, test_loader, criterion, checkpoint_path=None, verbose=True, fusion=False, num_samples=100):
    if checkpoint_path:
        best_model_path = checkpoint_path
    else:
        best_model_path = os.path.join(training_checkpoints, 'best_tabular.pth')  
    
    checkpoint = torch.load(best_model_path, weights_only=False) # load the best model checkpoint
    model.load_state_dict(checkpoint['model_state_dict']) # load saved weights
    model.eval()

    enable_dropout(model)
    
    mc_results = {}
    test_embeddings = {}
    
    for s in range(num_samples):
        entry_counter = 0
        # optionally show progress only on first pass for neatness
        if verbose and s == 0:
            loop_iter = tqdm(test_loader, desc=f'MC Pass {s+1}/{num_samples}')
        else:
            loop_iter = test_loader
        
        for continuous_data, labels, patient_id in loop_iter:
            continuous_data = continuous_data.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)
            
            # forward pass
            with torch.no_grad():
                embeddings, logits = model(continuous_data)
                probs = torch.sigmoid(logits).float()
            
            # store label, embeddings (once) and probabilities
            for pid, label_val, emb, pb, lg in zip(patient_id, labels, embeddings, probs, logits):
                pid_item = pid.item()
                entry_id = f"{pid_item}_{entry_counter}"
                
                if entry_id not in mc_results:
                    mc_results[entry_id] = {
                        'patient_id': pid_item,
                        'labels': int(label_val.item()),
                        'logits': [],
                        'probs': [],
                    }
                mc_results[entry_id]['logits'].append(lg.detach().cpu().item())
                mc_results[entry_id]['probs'].append(pb.detach().cpu().item())
                
                # Store embedding only once (on first pass)
                if s == 0:
                    if entry_id not in test_embeddings:
                        test_embeddings[entry_id] = []
                    test_embeddings[entry_id].append({
                        'patient_id': pid_item,
                        'label': int(label_val.item()),
                        'embedding': emb.detach().cpu(),
                    })
    
    # Now compute mean/std of the probabilities and decide final predictions
    test_results = []
    all_test_labels = []
    all_test_probs = []
    all_test_preds = []
            
    for entry_id, data in mc_results.items():
        label_val = data['labels']
        raw_probs = torch.tensor(data['probs'])
        mean_prob = raw_probs.mean().item()
        std_prob = raw_probs.std().item()
        pred = 1 if mean_prob > 0.5 else 0
        pid = data['patient_id']
        
        test_results.append((pid, label_val, pred, mean_prob, std_prob))
        all_test_labels.append(label_val)
        all_test_probs.append(mean_prob)
        all_test_preds.append(pred)

     # metrics
    test_total = len(all_test_labels)
    correct_preds = sum([1 for i in range(test_total) if all_test_labels[i] == all_test_preds[i]])
    test_accuracy = 100.0 * correct_preds / test_total
    test_auc = roc_auc_score(all_test_labels, all_test_probs)
    test_mcc = matthews_corrcoef(all_test_labels, all_test_preds)
    precision, recall, f1, _ = score(all_test_labels, all_test_preds, average='binary', zero_division=0)

    # log
    if not fusion:
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(f'Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')
        print(f'Test Precision: {precision:.4f}, Test Recall (Sensitivity): {recall:.4f}, Test F1-Score: {f1:.4f}')

        cm = confusion_matrix(all_test_labels, all_test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF","AFF"])
        disp.plot(cmap='Blues')
        plt.title('Tabular Image-Level Confusion Matrix (MC Dropout)')
        plt.show()

    # Save test results to DataFrame (including uncertainty)
    test_result_df = pd.DataFrame(test_results,
                                 columns=['patient_id', 'label', 'prediction', 'probability', 'uncertainty'])

    # Save out as in tabular_test
    if not fusion:
        if checkpoint_path:
            test_results_dir = os.path.dirname(checkpoint_path)
            test_res_path = os.path.join(test_results_dir, 'test_results_tab.pth')
        else:
            test_res_path = os.path.join(training_checkpoints, 'test_results_tab.pth')

    torch.save({'test_results': test_result_df,
                'test_embeddings': test_embeddings,
                }, test_res_path)

    aggregate_patient_results(test_result_df)

    return {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': test_auc,
        'mcc': test_mcc
    }

def aggregate_patient_results(test_df, inv_weight = True,fold_no=None):
    results = test_df.copy()
    
    if inv_weight:
        epsilon = 1e-6
        results['weight'] = 1 / (results['uncertainty'] + epsilon)
    else:
        results['weight'] = 1.0
        
    patient_results = (
        results.groupby('patient_id')
        .apply(lambda x: pd.Series({
            'label': x['label'].iloc[0],  # All labels are the same for each patient
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
    
    # calculate AUC and MCC for patient-level results
    patient_auc = roc_auc_score(patient_results['label'], patient_results['weighted_probability'])
    patient_mcc = matthews_corrcoef(patient_results['label'], patient_results['prediction'])
    
    print("\nPatient-level Metrics with MC Dropout:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC: {patient_auc:.4f}, MCC: {patient_mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save results
    os.makedirs(f'tabular_checkpoints/fold_{fold_no}', exist_ok=True)
    patient_results.to_csv(f'tabular_checkpoints/fold_{fold_no}/patient_results_mc_weighted.csv', index=False)
    
    # confusion matrix for patient-level predictions
    cm = confusion_matrix(patient_results['label'], patient_results['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Tabular Patient-Level Confusion Matrix (MC Dropout)')
    plt.savefig('tabular_checkpoints/patient_level_confusion_matrix_mc.png')
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
    plt.savefig('image_checkpoints/patient_level_uncertainty.png')
    plt.show()
    
    return patient_results

def main():
    final_df = preprocess_data()
    train_data, val_data, test_data = create_subset(final_df)
    _, _, test_loader = create_dataloaders(train_data, val_data, test_data)
    
    class_weights = compute_class_weights(train_data)
    
    # Load the best model checkpoint
    checkpoint = torch.load(training_checkpoints+best_model, weights_only=False)
    hyperparameters = checkpoint.get('hyperparameters', {})
    attention_dropout = hyperparameters.get('attention_dropout', 0.3)
    feedforward_dropout = hyperparameters.get('feedforward_dropout', 0.1)
    dimensions = hyperparameters.get('dimensions', 32)
    depth = hyperparameters.get('depth', 6)  
    heads = hyperparameters.get('heads', 8)
    
    model = TabularTransformer(
        attn_dropout=attention_dropout, 
        ff_dropout=feedforward_dropout,
        dim=dimensions,
        depth=depth,
        heads=heads,
        dropout_rate=0.5  # Added explicit dropout rate for MC dropout
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    test_model(model, test_loader, criterion)

if __name__ == "__main__":
    main()
# %%