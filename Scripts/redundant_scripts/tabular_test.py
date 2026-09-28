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
# create an empty tensor for categorical data 
#empty_categorical = torch.empty(0, 0).to(device)

# define tabular transformer model
class TabularTransformer(nn.Module):
    def __init__(self, attn_dropout=0.2, ff_dropout=0.2, dim=32, depth=6, heads=8):
        super(TabularTransformer, self).__init__()
        # for extracting later
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.model = FTTransformer(
            categories=(),              # empty tuple, since there are no categorical variables
            num_continuous=input_size,  # number of continuous variables
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
        
        categorical_data = torch.empty(x.size(0), 0) # empty tensor for categorical data
        continuous_data = x # continuous data
        
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

def test_model(model, test_loader, criterion, checkpoint_path=None, verbose = True, fusion = False):
    if checkpoint_path:
        best_model_path = checkpoint_path
    else:
        best_model_path = os.path.join(training_checkpoints, 'best_tabular.pth')  
    
    checkpoint = torch.load(best_model_path, weights_only=False) # load the best model checkpoint
    model.load_state_dict(checkpoint['model_state_dict']) # load saved weights
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
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
            
        for continuous_data, labels, patient_id in progress_testing:
            continuous_data = continuous_data.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float().view(-1, 1)

            outputs = model(continuous_data)
            logits = outputs[1]
            embeddings = outputs[0]
            loss = criterion(logits, labels)
            test_loss += loss.detach() * continuous_data.size(0)

            probs = torch.sigmoid(logits).float()
            predicted = (probs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().detach()

            # Store labels, probabilities and predictions for metrics calculation
            all_test_labels.extend(labels.detach().cpu().numpy().flatten())
            all_test_probs.extend(probs.detach().cpu().numpy().flatten())
            all_test_preds.extend(predicted.detach().cpu().numpy().flatten())
            
            for pid, label, pred, prob in zip(patient_id, labels, predicted, probs):
                    test_results.append((pid.item(), int(label.item()), int(pred.item()), float(prob.item())))
            
            for pid, label, emb in zip(patient_id, labels, embeddings):
                    pid_item = pid.item()
                    test_embeddings[pid_item] = {
                        'label' : int(label.item()),
                        'embedding' : emb.detach().cpu()#.numpy()
                    }
                
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
        disp.plot(cmap = 'Blues')
        plt.show()
    
    
    test_result_df = pd.DataFrame(test_results, columns=['patient_id', 'label', 'prediction', 'probability'])
    
    # save results to fold-specific path
    if not fusion:
        if checkpoint_path:
            test_results_dir = os.path.dirname(checkpoint_path)
            test_res_path = os.path.join(test_results_dir, 'test_results_tab.pth')
        else:
            test_res_path = os.path.join(training_checkpoints, 'test_results_tab.pth')

    torch.save({'test_results': test_result_df,
                'test_embeddings': test_embeddings,
                }, test_res_path)
    

    

    # for cross-balidation
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
        heads=heads
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    test_model(model, test_loader, criterion)

if __name__ == "__main__":
    main()
# %%