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
training_checkpoints = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'

# define vision transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=1):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=True, num_classes=num_classes)

        # output single unit logit
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
         
        embeddings = self.model.forward_features(x) # get features from the transformer
        embeddings = embeddings.mean(dim=[2, 3]) # average pooling of height and width to match the expected input dimensions
        preds = self.model.forward_head(embeddings)  # output the logits for predictions
        
        # output embeddings and predictions
        return embeddings, preds

def test_model(model, test_loader, criterion, fusion=False, verbose=True, checkpoint_path=None):
    
    if checkpoint_path:
        best_model_path = checkpoint_path
    else:
        best_model_path = os.path.join(training_checkpoints, 'best_vision.pth')       
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_labels = []
    test_predictions = []
    test_results = []
    test_embeddings = {}
    
    all_test_labels = []
    all_test_probs = []
    all_test_preds = []

    with torch.no_grad():
        
        if verbose:
            progress_testing = tqdm(test_loader, desc='Testing')
        else:
            progress_testing = test_loader
            
        for patient_id, images, labels in progress_testing:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(images)
            logits = outputs[1]
            embeddings = outputs[0]
            loss = criterion(logits, labels)
            test_loss += loss.detach() * images.size(0)

            probs = torch.sigmoid(logits).float()
            predicted = (probs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().detach()
            
            all_test_labels.extend(labels.detach().cpu().numpy().flatten())
            all_test_probs.extend(probs.detach().cpu().numpy().flatten())
            all_test_preds.extend(predicted.detach().cpu().numpy().flatten())

            for pid, label, pred, prob in zip(patient_id, labels, predicted, probs):
                    test_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
            
            for pid, label, emb in zip(patient_id, labels, embeddings):
                pid_item = pid.item()
                entry_id = f"{pid_item}_{len(test_embeddings.get(pid_item, []))}"
                if pid_item not in test_embeddings:
                    test_embeddings[pid_item] = []
                test_embeddings[pid_item].append({
                    'label': int(label.item()),
                    'embedding': emb.detach().cpu(),
                    'entry_id': entry_id
                })
            
            if verbose:
                progress_testing.set_postfix(loss=test_loss / test_total, accuracy=100 * test_correct / test_total)

    test_loss /= test_total
    test_accuracy = 100 * test_correct / test_total
    test_auc = roc_auc_score(all_test_labels, all_test_probs)
    test_mcc = matthews_corrcoef(all_test_labels, all_test_preds)

    precision, recall, f1, _ = score(all_test_labels, all_test_preds, average='binary', zero_division=0)

    if not fusion:
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        print(f'Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')
        print(f'Test Precision: {precision:.4f}, Test Recall (Sensitivity): {recall:.4f}, Test F1-Score: {f1:.4f}')

        # confusion matrix
        cm = confusion_matrix(all_test_labels, all_test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF","AFF"])
        disp.plot(cmap='Blues')
        plt.show()
        
    test_result_df = pd.DataFrame(test_results, columns=['patient_id', 'label', 'prediction', 'probability'])
    #test_result_df.to_csv('/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/test_results.csv', index=True)

    if not fusion:
        if checkpoint_path:
            test_results_dir = os.path.dirname(checkpoint_path)
            test_res_path = os.path.join(test_results_dir, 'test_results_img.pth')
        else:
            test_res_path = os.path.join(training_checkpoints, 'test_results_img.pth')
            
    torch.save({'test_results': test_result_df,
                'test_embeddings': test_embeddings,
                }, test_res_path)
    
    return {
        'loss': test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss,
        'accuracy': test_accuracy.item() if isinstance(test_accuracy, torch.Tensor) else test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': test_auc,
        'mcc': test_mcc
    }

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