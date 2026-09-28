# %%
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
import torch
from MLstatkit.stats import Delong_test

def compare_uncertainties_with_wilcoxon():
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


def compare_auc_with_delong():
    """
    Compare AUC scores between different models using DeLong test.
    The AUC is calculated as the average across all folds.
    Patient predictions are averaged across folds before performing the DeLong test.
    
    Returns:
        pd.DataFrame: DataFrame with p-values from DeLong tests
        pd.DataFrame: DataFrame with z-scores from DeLong tests
    """
    # Define paths for each modality
    tab_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'
    vis_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'
    late_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/'
    intra_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/intra_fusion_checkpoints/'
    inter_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/inter_fusion_checkpoints/'
    
    # Dictionary to store predictions and true labels for each modality by fold
    modality_data_by_fold = {}
    modalities = ['Tabular', 'Vision', 'Late', 'Intra', 'Inter']
    paths = [tab_path, vis_path, late_path, intra_path, inter_path]
    
    # Collect data for each modality across all folds separately
    for modality, path in zip(modalities, paths):
        modality_data_by_fold[modality] = []
        
        # Collect data from each fold separately
        for fold in range(1, 6):
            fold_name = f'fold_{fold}'
            df = pd.read_csv(os.path.join(path, fold_name, 'patient_results_mc_weighted.csv'))
            
            fold_data = {
                'true': df['label'].values,
                'prob': df['weighted_probability'].values
            }
            modality_data_by_fold[modality].append(fold_data)
    
    # Calculate AUCs for each modality and fold
    auc_scores = {modality: [] for modality in modalities}
    
    # Calculate AUC for each fold and modality
    for modality in modalities:
        for fold_data in modality_data_by_fold[modality]:
            auc = roc_auc_score(fold_data['true'], fold_data['prob'])
            auc_scores[modality].append(auc)
    
    # Calculate average AUC for each modality
    avg_auc_scores = {modality: np.mean(scores) for modality, scores in auc_scores.items()}
    std_auc_scores = {modality: np.std(scores) for modality, scores in auc_scores.items()}
    
    # Print the average AUC scores
    print("===== Average AUC Scores by Modality (across folds) =====")
    for modality in modalities:
        print(f"{modality}: {avg_auc_scores[modality]:.4f} ± {std_auc_scores[modality]:.4f}")
    print("\n")
    
    # Dictionary to store averaged predictions for each patient by modality
    patient_avg_predictions = {}
    
    # For each modality, average the predictions across folds for each patient
    for modality in modalities:
        # Get number of patients from the first fold's data
        n_patients = len(modality_data_by_fold[modality][0]['true'])
        
        # Initialize arrays for averaged predictions and true values
        avg_probs = np.zeros(n_patients)
        true_values = modality_data_by_fold[modality][0]['true']  # Truth values should be the same across folds
        
        # Sum probabilities across all folds
        for fold_idx in range(5):
            avg_probs += modality_data_by_fold[modality][fold_idx]['prob']
        
        # Divide by number of folds to get average
        avg_probs /= 5
        
        # Store the averaged predictions
        patient_avg_predictions[modality] = {
            'true': true_values,
            'prob': avg_probs
        }
    
    # Create result dataframes to store DeLong test results
    p_values = pd.DataFrame(index=modalities, columns=modalities)
    z_scores = pd.DataFrame(index=modalities, columns=modalities)
    
    # Perform DeLong test for all pairs of modalities using averaged predictions
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i == j:  # Same modality
                p_values.loc[mod1, mod2] = 1.0
                z_scores.loc[mod1, mod2] = 0.0
                continue
            
            # Get the averaged predictions for both modalities
            true_values = patient_avg_predictions[mod1]['true']
            avg_prob_mod1 = patient_avg_predictions[mod1]['prob']
            avg_prob_mod2 = patient_avg_predictions[mod2]['prob']
            
            # Perform DeLong test on the averaged predictions
            z_score, p_value = Delong_test(true_values, avg_prob_mod1, avg_prob_mod2)
            
            # Store results
            p_values.loc[mod1, mod2] = p_value
            z_scores.loc[mod1, mod2] = z_score
    
    # Print results
    print("===== DeLong Test P-values (Using Patient-Averaged Predictions) =====")
    print(p_values.round(6))
    print("\n===== DeLong Test Z-scores (Using Patient-Averaged Predictions) =====")
    print(z_scores.round(4))
    
    # Calculate Bonferroni correction factor - number of pairwise comparisons
    n_modalities = len(modalities)
    n_comparisons = n_modalities * (n_modalities - 1) / 2
    alpha_corrected = 0.05 / 6
    
    # List all comparisons with Bonferroni correction
    print(f"\n===== All Pairwise Comparisons (Bonferroni-corrected α = {alpha_corrected:.6f}) =====")
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i < j:  # Print each pair only once
                p = p_values.loc[mod1, mod2]
                auc1 = avg_auc_scores[mod1]
                auc2 = avg_auc_scores[mod2]
                direction = "higher" if auc1 > auc2 else "lower"
                significance = "significant" if p < alpha_corrected else "not significant"
                # Format very small p-values in scientific notation
                if p < 0.000001:
                    p_str = f"{p:.2e}"
                else:
                    p_str = f"{p:.6f}"
                print(f"{mod1} vs {mod2}: p={p_str} ({mod1} has {direction} AUC, {auc1:.4f} vs {auc2:.4f}) - {significance}")
    
    # Also show a summary of only significant differences after correction
    print("\n===== Significant Differences after Bonferroni Correction =====")
    significant_count = 0
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i < j:  # Check each pair only once
                p = p_values.loc[mod1, mod2]
                if p < alpha_corrected:
                    auc1 = avg_auc_scores[mod1]
                    auc2 = avg_auc_scores[mod2]
                    direction = "higher" if auc1 > auc2 else "lower"
                    # Format very small p-values in scientific notation
                    if p < 0.000001:
                        p_str = f"{p:.2e}"
                    else:
                        p_str = f"{p:.6f}"
                    print(f"{mod1} vs {mod2}: p={p_str} ({mod1} has {direction} AUC, {auc1:.4f} vs {auc2:.4f})")
                    significant_count += 1
    
    if significant_count == 0:
        print("No significant differences after Bonferroni correction.")
    
    return p_values, z_scores

def calculate_mean_prediction_aucs():
    """
    Calculate AUC scores using patient-averaged predictions across folds.
    This provides AUC scores that directly correspond to the data used in the DeLong test.
    """
    # Define paths for each modality
    tab_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'
    vis_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'
    late_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/'
    intra_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/intra_fusion_checkpoints/'
    inter_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/inter_fusion_checkpoints/'
    
    modalities = ['Tabular', 'Vision', 'Late', 'Intra', 'Inter']
    paths = [tab_path, vis_path, late_path, intra_path, inter_path]
    
    # Dictionary to store AUCs calculated on patient-averaged predictions
    mean_prediction_aucs = {}
    
    # Process each modality
    for modality, path in zip(modalities, paths):
        # Initialize arrays for patient predictions
        patient_probs = []
        patient_labels = None
        
        # Read data from all folds
        for fold in range(1, 6):
            fold_name = f'fold_{fold}'
            df = pd.read_csv(os.path.join(path, fold_name, 'patient_results_mc_weighted.csv'))
            
            # Store probabilities from this fold
            patient_probs.append(df['weighted_probability'].values)
            
            # Store the labels (same in each fold)
            if patient_labels is None:
                patient_labels = df['label'].values
        
        # Convert to numpy array for easier averaging
        patient_probs = np.array(patient_probs)
        
        # Average probabilities across folds
        avg_probs = np.mean(patient_probs, axis=0)
        
        # Calculate AUC on patient-averaged predictions
        mean_prediction_auc = roc_auc_score(patient_labels, avg_probs)
        mean_prediction_aucs[modality] = mean_prediction_auc
    
    # Print results
    print("===== AUC Scores on Patient-Averaged Predictions =====")
    for modality in modalities:
        print(f"{modality}: {mean_prediction_aucs[modality]:.4f}")
    
    return mean_prediction_aucs

# Add this to your main() function:
# mean_aucs = calculate_mean_prediction_aucs()

def compare_avg_uncertainties_with_wilcoxon():
    """
    Compare model uncertainties using Wilcoxon signed-rank test on patient-averaged uncertainties.
    This maintains statistical validity by ensuring each patient is represented only once in the test.
    
    Returns:
        pd.DataFrame: DataFrame with p-values from Wilcoxon tests
        pd.DataFrame: DataFrame with test statistics from Wilcoxon tests
    """
    # Define paths for each modality
    tab_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'
    vis_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'
    late_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/late_fusion_checkpoints/'
    intra_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/intra_fusion_checkpoints/'
    inter_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/inter_fusion_checkpoints/'
    
    # Load patient uncertainty data for each modality
    tab_unc = pd.read_csv(os.path.join(tab_path, 'patient_uncertainty_averaged.csv'))
    vis_unc = pd.read_csv(os.path.join(vis_path, 'patient_uncertainty_averaged.csv'))
    late_unc = pd.read_csv(os.path.join(late_path, 'patient_uncertainty_averaged.csv'))
    intra_unc = pd.read_csv(os.path.join(intra_path, 'patient_uncertainty_averaged.csv'))
    inter_unc = pd.read_csv(os.path.join(inter_path, 'patient_uncertainty_averaged.csv'))
    
    # Extract patient-averaged uncertainties
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
    
    # Display median and mean values for each modality
    print("===== Uncertainty Statistics by Modality =====")
    for name, values in uncertainties.items():
        print(f"{name}: Mean = {np.mean(values):.6f}, Median = {np.median(values):.6f}, Std = {np.std(values):.6f}")
    print("\n")
    
    # Perform Wilcoxon tests for all pairs of modalities
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i == j:  # Same modality
                p_values.loc[mod1, mod2] = 1.0
                statistics.loc[mod1, mod2] = 0.0
                continue
                
            # Perform Wilcoxon signed-rank test
            stat, p = wilcoxon(uncertainties[mod1], uncertainties[mod2], 
                              alternative='two-sided')
            p_values.loc[mod1, mod2] = p
            statistics.loc[mod1, mod2] = stat
    
    # Print results
    print("===== Wilcoxon Signed-Rank Test P-values =====")
    print(p_values.round(6))
    print("\n===== Wilcoxon Signed-Rank Test Statistics =====")
    print(statistics.round(2))
    
    # Calculate Bonferroni correction factor - number of pairwise comparisons
    n_modalities = len(modalities)
    n_comparisons = n_modalities * (n_modalities - 1) / 2
    alpha_corrected = 0.05 / n_comparisons
    
    # List all comparisons with Bonferroni correction
    print(f"\n===== All Pairwise Comparisons (Bonferroni-corrected α = {alpha_corrected:.6f}) =====")
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i < j:  # Print each pair only once
                p = p_values.loc[mod1, mod2]
                med1 = np.median(uncertainties[mod1])
                med2 = np.median(uncertainties[mod2])
                direction = "lower" if med1 < med2 else "higher"
                significance = "significant" if p < alpha_corrected else "not significant"
                # Format very small p-values in scientific notation
                if p < 0.000001:
                    p_str = f"{p:.2e}"
                else:
                    p_str = f"{p:.6f}"
                print(f"{mod1} vs {mod2}: p={p_str} ({mod1} has {direction} uncertainty, {med1:.6f} vs {med2:.6f}) - {significance}")
    
    # Also show a summary of only significant differences after correction
    print("\n===== Significant Differences after Bonferroni Correction =====")
    significant_count = 0
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i < j:  # Check each pair only once
                p = p_values.loc[mod1, mod2]
                if p < alpha_corrected:
                    med1 = np.median(uncertainties[mod1])
                    med2 = np.median(uncertainties[mod2])
                    direction = "lower" if med1 < med2 else "higher"
                    # Format very small p-values in scientific notation
                    if p < 0.000001:
                        p_str = f"{p:.2e}"
                    else:
                        p_str = f"{p:.6f}"
                    print(f"{mod1} vs {mod2}: p={p_str} ({mod1} has {direction} uncertainty, {med1:.6f} vs {med2:.6f})")
                    significant_count += 1
    
    if significant_count == 0:
        print("No significant differences after Bonferroni correction.")
    
    return p_values, statistics

def main():
    """
    #p_vals, stats = compare_uncertainties_with_wilcoxon()
    print("\n\n===== Comparing AUCs using DeLong test =====")
    delong_p_vals, delong_z_scores = compare_auc_with_delong()
    
    # Print the p-values in scientific notation to better display very small values
    print("\n===== DeLong Test P-values (Scientific Notation) =====")
    pd.options.display.float_format = '{:.2e}'.format
    print(delong_p_vals)
    
    mean_aucs = calculate_mean_prediction_aucs()
    """
    
    """
    print("\n\n===== Comparing Model Uncertainties using Wilcoxon test =====")
    uncertainty_p_vals, uncertainty_stats = compare_avg_uncertainties_with_wilcoxon()
    
    # Print the p-values in scientific notation
    print("\n===== Wilcoxon Test P-values (Scientific Notation) =====")
    pd.options.display.float_format = '{:.2e}'.format
    print(uncertainty_p_vals)
     """
     
     
    checkpoint = torch.load("fusion_pipeline/fold_3/best_vision_model.pth", weights_only=False)
    #print("Checkpoint keys:", list(checkpoint.keys()))
    hyperparameters = checkpoint["hyperparameters"]  # or similar key depending on how it's stored
    print(hyperparameters)
if __name__ == '__main__':
    main()
# %%