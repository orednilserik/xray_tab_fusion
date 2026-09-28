# %%
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support as score
from tqdm import tqdm
import os
import torch.nn as nn
import timm
from vision_train import *
from sklearn.metrics import roc_auc_score, matthews_corrcoef 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_checkpoints = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/fusion_pipeline/'

# define vision transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model('tiny_vit_5m_224.dist_in22k_ft_in1k', pretrained=True, num_classes=num_classes)
        
        # explicit dropout for MC Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # output single unit logit
        self.model.head = nn.Sequential(
            self.dropout,
            nn.Linear(self.model.head.in_features, 1)
        )
        
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
         
        embeddings = self.model.forward_features(x) # get features from the transformer
        embeddings = embeddings.mean(dim=[2, 3]) # average pooling of height and width to match the expected input dimensions
        preds = self.model.forward_head(embeddings)  # output the logits for predictions
        
        # output embeddings and predictions
        return embeddings, preds

def enable_dropout(model):
    # enable dropout layers for MC dropout inference
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # set dropout layers to training mode during inference

def test_model(model, test_loader, criterion, fusion=False, verbose=True, checkpoint_path=None, num_samples=100):
    
    if checkpoint_path:
        best_model_path = checkpoint_path
    else:
        best_model_path = os.path.join(training_checkpoints, 'best_vision.pth')       
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    enable_dropout(model)
    
    mc_results = {}
    test_embeddings = {}  # same structure as in vision_test
    
    for s in range(num_samples):
        entry_counter = 0
        # optionally show progress only on first pass for neatness
        if verbose and s == 0:
            loop_iter = tqdm(test_loader, desc=f'MC Pass {s+1}/{num_samples}')
        else:
            loop_iter = test_loader

        for patient_id, images, labels in loop_iter:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            # forward pass
            with torch.no_grad():
                embeddings, logits = model(images)
                probs = torch.sigmoid(logits).float()

            # store label, embeddings and probabilities
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
            
                # store embedding only once (on first pass)
                if s == 0:
                    if entry_id not in test_embeddings:
                        test_embeddings[entry_id] = []
                    test_embeddings[entry_id].append({
                        'patient_id': pid_item,
                        'label': int(label_val.item()),
                        'embedding': emb.detach().cpu(),
                    })
            
                entry_counter += 1
    # compute mean+std of the probabilities and decide final predictions
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
        plt.title('Vision Image-Level Confusion Matrix (MC Dropout)')
        plt.show()
        

    # Save test results to DataFrame (including uncertainty if you like)
    test_result_df = pd.DataFrame(test_results,
                                  columns=['patient_id', 'label', 'prediction', 'probability', 'uncertainty'])

    # Save out as in vision_test
    if not fusion:
        if checkpoint_path:
            test_results_dir = os.path.dirname(checkpoint_path)
            test_res_path = os.path.join(test_results_dir, 'test_results_img.pth')
        else:
            test_res_path = os.path.join(training_checkpoints, 'test_results_img.pth')

    torch.save({
        'test_results': test_result_df,
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
    
    os.makedirs(f'image_checkpoints/fold_{fold_no}', exist_ok=True)
    
    # Save results
    patient_results.to_csv(f'image_checkpoints/fold_{fold_no}/patient_results_mc_weighted.csv', index=False)
    
    # confusion matrix for patient-level predictions
    cm = confusion_matrix(patient_results['label'], patient_results['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Vision Patient-Level Confusion Matrix (MC Dropout)')
    plt.savefig('image_checkpoints/patient_level_confusion_matrix_mc.png')
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
    # load the saved test data and class weights
    test_loader = torch.load(training_checkpoints + 'image_test_loader.pth', weights_only=False)
    class_weights = torch.load(training_checkpoints + 'class_weights.pth', weights_only=False)
    
    model = VisionTransformer(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    test_model(model, test_loader, criterion)

if __name__ == "__main__":
    main()
# %%