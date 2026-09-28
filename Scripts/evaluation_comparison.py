# %%
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from MLstatkit.stats import Delong_test
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, f1_score, matthews_corrcoef

def plot_roc_curve(true_y, y_prob, label=None):
    fpr, tpr, _ = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

def plot_cv_roc_tab(fold_root='fusion_pipeline/fold_*'):
    """
    Plot per‐fold ROC curves and their mean.
    Assumes each fold dir has:
      - best_tabular_model.pth   (with 'epoch')
      - test_results_tab.pth     (with DataFrame 'test_results')
    """
    fold_dirs = sorted(glob.glob(fold_root))
    if not fold_dirs:
        print(f"No fold directories found under pattern {fold_root}")
        return

    plt.figure(figsize=(10, 6))
    interp_rocs = []
    for fd in fold_dirs:
        fold_no = os.path.basename(fd).split('_')[-1]
        # load test results
        test_pth = os.path.join(fd, 'test_results_tab.pth')
        if not os.path.exists(test_pth):
            continue
        df = torch.load(test_pth, weights_only=False)['test_results']
        y_true = df['label'].values
        y_prob = df['probability'].values

        # compute ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # load epoch info
        ckpt = torch.load(os.path.join(fd, 'best_tabular_model.pth'), weights_only=False)
        epoch = ckpt.get('epoch', None)

        label = f"Fold {fold_no} (Epoch {epoch}), AUC {roc_auc:.3f}"
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=label)

        interp_rocs.append((fpr, tpr))

    # mean ROC
    all_fpr = np.unique(np.concatenate([f for f, _ in interp_rocs]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in interp_rocs:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(interp_rocs)
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--',
             lw=2, label=f"Mean ROC — AUC {mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross‐Validated ROC Curves (Tabular)')
    plt.legend(loc='lower right')
    os.makedirs('fusion_pipeline', exist_ok=True)
    plt.savefig('fusion_pipeline/cv_tabular_roc.png')
    plt.show()
    
def plot_cv_roc_img(fold_root='fusion_pipeline/fold_*'):
    """
    Plot per‐fold ROC curves and their mean.
    Assumes each fold dir has:
      - best_vision_model.pth   (with 'epoch')
      - test_results_img.pth     (with DataFrame 'test_results')
    """
    fold_dirs = sorted(glob.glob(fold_root))
    if not fold_dirs:
        print(f"No fold directories found under pattern {fold_root}")
        return

    plt.figure(figsize=(10,6))
    interp_rocs = []
    for fd in fold_dirs:
        fold_no = os.path.basename(fd).split('_')[-1]
        # load test results
        test_pth = os.path.join(fd, 'test_results_img.pth')
        if not os.path.exists(test_pth):
            continue
        df = torch.load(test_pth, weights_only=False)['test_results']
        y_true = df['label'].values
        y_prob = df['probability'].values

        # compute ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # load epoch info
        ckpt = torch.load(os.path.join(fd, 'best_vision_model.pth'), weights_only=False)
        epoch = ckpt.get('epoch', None)

        label = f"Fold {fold_no} (Epoch {epoch}), AUC {roc_auc:.3f}"
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=label)

        interp_rocs.append((fpr, tpr))

    # mean ROC
    all_fpr = np.unique(np.concatenate([f for f, _ in interp_rocs]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in interp_rocs:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(interp_rocs)
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--',
             lw=2, label=f"Mean ROC — AUC {mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cross‐Validated ROC Curves (Vision)')
    plt.legend(loc='lower right')
    os.makedirs('fusion_pipeline', exist_ok=True)
    plt.savefig('fusion_pipeline/cv_vision_roc.png')
    plt.show()

def plot_cv_roc_fusion_agg(fold_root='inter_fusion_checkpoints/fold_*',fusion_type='Inter'):

    fold_dirs = sorted(glob.glob(fold_root))
    if not fold_dirs:
        print(f"No fold directories found under pattern {fold_root}")
        return

    plt.figure(figsize=(10,6))
    interp_rocs = []
    for fd in fold_dirs:
        fold_no = os.path.basename(fd).split('_')[-1]
        # load test results
        test_csv = os.path.join(fd, 'patient_results_mc_weighted.csv')
        if not os.path.exists(test_csv):
            continue
        df = pd.read_csv(test_csv)
        y_true = df['label'].values
        y_prob = df['weighted_probability'].values

        # compute ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # load epoch info
        #ckpt = torch.load(os.path.join(fd, 'best_vision_model.pth'), weights_only=False)
        #epoch = ckpt.get('epoch', None)

        label = f"Fold {fold_no}, AUC {roc_auc:.3f}"
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=label)

        interp_rocs.append((fpr, tpr))

    # mean ROC
    all_fpr = np.unique(np.concatenate([f for f, _ in interp_rocs]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in interp_rocs:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(interp_rocs)
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--',
             lw=2, label=f"Mean ROC — AUC {mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cross‐Validated ROC Curves ({fusion_type}) Patient Level')
    plt.legend(loc='lower right')
    os.makedirs('fusion_pipeline', exist_ok=True)
    plt.savefig('fusion_pipeline/cv_vision_roc_agg.png')
    plt.show()




    # mean ROC
    all_fpr = np.unique(np.concatenate([f for f, _ in interp_rocs]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in interp_rocs:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(interp_rocs)
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--',
             lw=2, label=f"Mean ROC — AUC {mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cross‐Validated ROC Curves ({fusion_type}) Patient Level')
    plt.legend(loc='lower right')
    os.makedirs('fusion_pipeline', exist_ok=True)
    plt.savefig('fusion_pipeline/cv_vision_roc_agg.png')
    plt.show()

def plot_cv_roc_fusion(fold_root='fusion_pipeline/fold_*', fusion_type='Inter'):

    fold_dirs = sorted(glob.glob(fold_root))
    if not fold_dirs:
        print(f"No fold directories found under pattern {fold_root}")
        return

    plt.figure(figsize=(10,6))
    interp_rocs = []
    for fd in fold_dirs:
        fold_no = os.path.basename(fd).split('_')[-1]
        # load test results
        test_csv = os.path.join(fd, 'patient_results_mc_weighted.csv')
        if not os.path.exists(test_csv):
            continue
        df = pd.read_csv(test_csv)
        y_true = df['label'].values
        y_prob = df['weighted_probability'].values

        # compute ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        # load epoch info
        ckpt = torch.load(os.path.join(fd, 'best_fusion.pth'), weights_only=False)
        epoch = ckpt.get('epoch', None)

        label = f"Fold {fold_no} (Epoch {epoch}), AUC {roc_auc:.3f}"
        plt.plot(fpr, tpr, lw=1, alpha=0.6, label=label)

        interp_rocs.append((fpr, tpr))

    # mean ROC
    all_fpr = np.unique(np.concatenate([f for f, _ in interp_rocs]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in interp_rocs:
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(interp_rocs)
    mean_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--',
             lw=2, label=f"Mean ROC — AUC {mean_auc:.3f}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Cross‐Validated ROC Curves ({fusion_type})')
    plt.legend(loc='lower right')
    os.makedirs('fusion_pipeline', exist_ok=True)
    plt.savefig('fusion_pipeline/cv_vision_roc.png')
    plt.show()

def calculate_metrics_stats(modality='tabular', aggregate=False, all_folds=True, fold_no='fold_5'):
    """
    Calculate performance metrics for a model.
    
    Args:
        modality: 'tabular' or 'vision'
        aggregate: Whether to use patient-level aggregated results
        all_folds: Whether to calculate metrics across all folds (True) or just one fold (False)
        fold_no: Specific fold to analyze if all_folds=False
    """
    # Lists to store metrics from each fold
    aucs = []
    accuracies = []
    f1_scores = []
    mccs = []
    
    folds_to_process = []
    if all_folds:
        # Process all folds (assuming 5-fold CV)
        folds_to_process = [f'fold_{i}' for i in range(1, 6)]
    else:
        # Process only the specified fold
        folds_to_process = [fold_no]
    
    # Process each fold
    for fold in folds_to_process:
        if modality == 'tabular' and aggregate:
            checkpoint = f'/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/{fold}'
            df = pd.read_csv(os.path.join(checkpoint, 'patient_results_mc_weighted.csv'))
        
        if modality == 'vision' and aggregate:
            checkpoint = f'/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/{fold}'
            df = pd.read_csv(os.path.join(checkpoint, 'patient_results_mc_weighted.csv'))
            
        if not aggregate:
            checkpoint = f'/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/fusion_pipeline/{fold}'
        
            if modality == 'tabular':
                df_path = torch.load(os.path.join(checkpoint, 'test_results_tab.pth'), weights_only=False)
                df = df_path['test_results']
                
            elif modality == 'vision':
                df_path = torch.load(os.path.join(checkpoint, 'test_results_img.pth'), weights_only=False)
                df = df_path['test_results']
        
        if modality == 'intra':
            checkpoint = f'/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/intra_fusion_checkpoints/{fold}'
            
            if aggregate:
                df = pd.read_csv(os.path.join(checkpoint, 'patient_results_mc_weighted.csv'))
            else:
                df = pd.read_csv(os.path.join(checkpoint, 'test_results.csv'))
                
        if modality == 'inter':
            checkpoint = f'/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/inter_fusion_checkpoints/{fold}'
            
            if aggregate:
                df = pd.read_csv(os.path.join(checkpoint, 'patient_results_mc_weighted.csv'))
            else:
                df = pd.read_csv(os.path.join(checkpoint, 'test_results.csv'))
                
        if modality == 'late':
            checkpoint = f'/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/{fold}'
            
            if aggregate:
                df = pd.read_csv(os.path.join(checkpoint, 'patient_results_mc_weighted.csv'))
            else:
                df = pd.read_csv(os.path.join(checkpoint, 'test_results.csv'))
        
        y_true = df['label'].values
        if aggregate:
            y_prob = df['weighted_probability'].values
        elif not aggregate:
            y_prob = df['probability'].values
            
        y_pred = (y_prob >= 0.5).astype(int)  # Binary prediction using 0.5 threshold
            
        # Calculate metrics
        auc_score = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
            
        # Store metrics
        aucs.append(auc_score)
        accuracies.append(acc)
        f1_scores.append(f1)
        mccs.append(mcc)
            
        print(f"{fold}: ACC={acc:.4f}, AUC={auc_score:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
            
    # Calculate mean and std for each metric
    metrics = {
        "Accuracy": {"mean": np.mean(accuracies), "std": np.std(accuracies)},
        "AUC": {"mean": np.mean(aucs), "std": np.std(aucs)},
        "F1": {"mean": np.mean(f1_scores), "std": np.std(f1_scores)},
        "MCC": {"mean": np.mean(mccs), "std": np.std(mccs)}
    }
    
    # Print summary
    print("\n===== Metrics Summary =====")
    for metric, values in metrics.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    return metrics



import pandas as pd
import os
import numpy as np

from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

def compare_uncertainties_with_wilcoxon():
    """
    Perform Wilcoxon signed-rank tests between all pairs of uncertainty modalities
    and display results in a matrix format.
    """
    # Load uncertainty data
    tab_unc_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'
    vis_unc_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'
    late_unc_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/'
    intra_unc_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/intra_fusion_checkpoints/'
    inter_unc_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/inter_fusion_checkpoints/'
    
    tab_unc = pd.read_csv(os.path.join(tab_unc_path, 'patient_uncertainty_averaged.csv'))
    vis_unc = pd.read_csv(os.path.join(vis_unc_path, 'patient_uncertainty_averaged.csv'))
    late_unc = pd.read_csv(os.path.join(late_unc_path, 'patient_uncertainty_averaged.csv'))
    intra_unc = pd.read_csv(os.path.join(intra_unc_path, 'patient_uncertainty_averaged.csv'))
    inter_unc = pd.read_csv(os.path.join(inter_unc_path, 'patient_uncertainty_averaged.csv'))
    
    # Extract uncertainty values
    uncertainties = {
        'Tabular': tab_unc['avg_mean_uncertainty'].values,
        'Vision': vis_unc['avg_mean_uncertainty'].values,
        'Late': late_unc['avg_mean_uncertainty'].values,
        'Intra': intra_unc['avg_mean_uncertainty'].values,
        'Inter': inter_unc['avg_mean_uncertainty'].values
    }
    
    # Prepare results dataframes
    modalities = list(uncertainties.keys())
    p_values = pd.DataFrame(index=modalities, columns=modalities)
    statistics = pd.DataFrame(index=modalities, columns=modalities)
    
    # Display median values for each modality
    print("===== Median Uncertainty by Modality =====")
    for name, values in uncertainties.items():
        print(f"{name}: {np.median(values):.6f}")
    print("\n")
    
    # Perform Wilcoxon tests for all pairs
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i == j:  # Same modality
                p_values.loc[mod1, mod2] = 1.0
                statistics.loc[mod1, mod2] = 0.0
                continue
                
            # Perform Wilcoxon signed-rank test
            stat, p = wilcoxon(uncertainties[mod1], uncertainties[mod2])
            p_values.loc[mod1, mod2] = p
            statistics.loc[mod1, mod2] = stat
    
    # Print results
    print("===== Wilcoxon Signed-Rank Test P-values =====")
    print(p_values.round(6))
    print("\n===== Wilcoxon Signed-Rank Test Statistics =====")
    print(statistics.round(2))
    
    # Summarize significant differences (p < 0.05)
    print("\n===== Significant Differences (p < 0.05) =====")
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i < j:  # Print each pair only once
                p = p_values.loc[mod1, mod2]
                if p < 0.05/9:
                    med1 = np.median(uncertainties[mod1])
                    med2 = np.median(uncertainties[mod2])
                    direction = "lower" if med1 < med2 else "higher"
                    print(f"{mod1} vs {mod2}: p={p:.6f} ({mod1} has {direction} uncertainty)")
    
    return p_values, statistics

# Add this to your main function
# p_vals, stats = compare_uncertainties_with_wilcoxon()

def plot_combined_roc_curves(level='image',use_log_scale=True):

    plt.figure(figsize=(7, 4))
    
    # Define models and their paths based on level
    if level == 'image':
        models = {
            'Tabular': {'fold_pattern': 'fusion_pipeline/fold_*', 'result_file': 'test_results_tab.pth', 'color': 'blue'},
            'Vision': {'fold_pattern': 'fusion_pipeline/fold_*', 'result_file': 'test_results_img.pth', 'color': 'green'},
            'Intra': {'fold_pattern': 'intra_fusion_checkpoints/fold_*', 'result_file': 'test_results.csv', 'color': 'red'},
            'Inter': {'fold_pattern': 'inter_fusion_checkpoints/fold_*', 'result_file': 'test_results.csv', 'color': 'purple'},
            'Late': {'fold_pattern': 'late_fusion_checkpoints/fold_*', 'result_file': 'test_results.csv', 'color': 'orange'}
        }
    else:  # patient level
        models = {
            'Tabular': {'fold_pattern': 'tabular_checkpoints/fold_*', 'result_file': 'patient_results_mc_weighted.csv', 'color': 'blue'},
            'Vision': {'fold_pattern': 'image_checkpoints/fold_*', 'result_file': 'patient_results_mc_weighted.csv', 'color': 'green'},
            'Intra': {'fold_pattern': 'intra_fusion_checkpoints/fold_*', 'result_file': 'patient_results_mc_weighted.csv', 'color': 'red'},
            'Inter': {'fold_pattern': 'inter_fusion_checkpoints/fold_*', 'result_file': 'patient_results_mc_weighted.csv', 'color': 'purple'},
            'Late': {'fold_pattern': 'late_fusion_checkpoints/fold_*', 'result_file': 'patient_results_mc_weighted.csv', 'color': 'orange'}
        }
    
    # Plot ROC curve for each model
    for model_name, config in models.items():
        fold_dirs = sorted(glob.glob(config['fold_pattern']))
        if not fold_dirs:
            print(f"No fold directories found for {model_name} under pattern {config['fold_pattern']}")
            continue
            
        interp_rocs = []
        for fd in fold_dirs:
            fold_no = os.path.basename(fd).split('_')[-1]
            
            # Load test results based on file type
            result_path = os.path.join(fd, config['result_file'])
            if not os.path.exists(result_path):
                continue
                
            if result_path.endswith('.pth'):
                df = torch.load(result_path, weights_only=False)['test_results']
            else:  # CSV file
                df = pd.read_csv(result_path)
                
            # Extract labels and probabilities
            y_true = df['label'].values
            y_prob = df['probability'].values if 'probability' in df.columns else df['weighted_probability'].values
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            interp_rocs.append((fpr, tpr))
            
        if not interp_rocs:
            continue
            
        # Calculate mean ROC curve
        all_fpr = np.unique(np.concatenate([f for f, _ in interp_rocs]))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in interp_rocs:
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(interp_rocs)
        mean_auc = auc(all_fpr, mean_tpr)
        
        # Plot mean ROC curve
        plt.plot(all_fpr, mean_tpr, color=config['color'], linestyle='-',
                 lw=2, label=f"{model_name} (AUC: {mean_auc:.3f})")
    
    # Add diagonal reference line
    #plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
    if use_log_scale and level == 'image':
        # For log scale, we need to create points that will look like a diagonal
        # line when plotted in log-log space
        log_points = np.logspace(-4, 0, 100)  # Create points in log space
        plt.plot(log_points, log_points, color='gray', linestyle=':', label='Random classifier')
        
        # Set log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-3, 1.05)
        plt.ylim(1e-1, 1.05)
        plt.legend(loc='lower left', fontsize=8)
        
    elif use_log_scale and level == 'patient':
        log_points = np.logspace(-4, 0, 100)  # Create points in log space
        plt.plot(log_points, log_points, color='gray', linestyle=':', label='Random classifier')
        
        # Set log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-3, 1.0)
        plt.ylim(9e-1, 1.003)
        plt.legend(loc='best', fontsize=8, bbox_to_anchor=(0.95, 0.35))
        
    else:
        # Regular linear scale diagonal
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random classifier')
        plt.xlim(-0.01, 1.02)
        plt.ylim(0, 1.02)
        plt.legend(loc='lower right', fontsize=8)
    # Add labels and legend
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.grid(alpha=0.3)
    
    # Save figure
    os.makedirs('fusion_pipeline', exist_ok=True)
    plt.savefig(f'fusion_pipeline/combined_roc_{level}_level.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_all_models_metrics(aggregate=True):
    """
    Print a formatted table of performance metrics (Accuracy, AUC, F1, MCC) for all models.
    
    Args:
        aggregate: Whether to use patient-level aggregated results (True) or image-level results (False)
    """
    # Define all models to evaluate
    models = ['tabular', 'vision', 'intra', 'inter', 'late']
    
    # Dictionary to store results for each model
    all_metrics = {}
    
    # Calculate metrics for each model
    for model in models:
        print(f"\n===== Evaluating {model.capitalize()} Model =====")
        metrics = calculate_metrics_stats(modality=model, aggregate=aggregate)
        all_metrics[model.capitalize()] = metrics
    
    # Create formatted table header
    level = "Patient" if aggregate else "Image"
    print(f"\n========== {level}-Level Performance Metrics ==========")
    print(f"{'Model':<10} | {'Accuracy':<15} | {'AUC':<15} | {'F1':<15} | {'MCC':<15}")
    print("-" * 76)
    
    # Print metrics for each model in the table
    for model_name, metrics in all_metrics.items():
        acc = f"{metrics['Accuracy']['mean']:.4f} ± {metrics['Accuracy']['std']:.4f}"
        auc = f"{metrics['AUC']['mean']:.4f} ± {metrics['AUC']['std']:.4f}"
        f1 = f"{metrics['F1']['mean']:.4f} ± {metrics['F1']['std']:.4f}"
        mcc = f"{metrics['MCC']['mean']:.4f} ± {metrics['MCC']['std']:.4f}"
        
        print(f"{model_name:<10} | {acc:<15} | {auc:<15} | {f1:<15} | {mcc:<15}")
    
    return all_metrics

# %%
def main():
    plot_combined_roc_curves(level='image',use_log_scale=False)
    plot_combined_roc_curves(level='patient',use_log_scale=False)
    plot_combined_roc_curves(level='image',use_log_scale=True)
    plot_combined_roc_curves(level='patient',use_log_scale=True)
   # p_vals, stats = compare_uncertainties_with_wilcoxon()
    # Print metrics tables for both patient and image levels
    print("\n# Patient-Level Metrics")
    patient_metrics = print_all_models_metrics(aggregate=True)
    
    print("\n# Image-Level Metrics")
    image_metrics = print_all_models_metrics(aggregate=False)
    
if __name__ == '__main__':
    main()
    
# %%