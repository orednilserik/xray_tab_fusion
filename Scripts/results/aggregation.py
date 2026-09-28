# %%
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
from scipy.stats import ranksums

def aggregate_patient_results(input_dir, output_file, fold_pattern="fold_*_patient_results_img.csv"):
    """
    Aggregates patient results across multiple folds and computes averages.
    
    Args:
        input_dir: Directory containing the fold CSV files
        output_file: Path to save the aggregated results
        fold_pattern: Pattern to match fold CSV files
    """
    # Get absolute path if not already
    input_dir = os.path.abspath(input_dir)
    
    # Get all matching CSV files
    csv_files = glob.glob(os.path.join(input_dir, fold_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{fold_pattern}' in '{input_dir}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to aggregate")
    
    # Read and combine all CSVs
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        fold_name = os.path.basename(file).split('_')[1]
        df['fold'] = fold_name
        dfs.append(df)
    
    # Concatenate all dataframes
    all_data = pd.concat(dfs, ignore_index=True)
    
    # Group by patient_id and calculate averages
    grouped = all_data.groupby('patient_id').agg({
        'label': 'first',  # Assuming label is the same across folds
        'weighted_probability': 'mean',
        'mean_uncertainty': 'mean'
        #'num_patches': 'mean'  # Include this if you want average patch count
    }).reset_index()
    
    # Rename columns
    grouped = grouped.rename(columns={
        'weighted_probability': 'avg_weighted_probability',
        'mean_uncertainty': 'avg_patient_uncertainty',
        'num_patches': 'avg_num_patches'
    })
    
    # Add prediction column
    grouped['prediction'] = (grouped['avg_weighted_probability'] > 0.5).astype(int)
    
    # Save to CSV
    grouped.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")
    
    # Calculate and print some statistics
    accuracy = (grouped['prediction'] == grouped['label']).mean()
    print(f"Overall accuracy: {accuracy:.4f}")
    
    try:
        # Calculate ROC AUC
        auc = roc_auc_score(grouped['label'], grouped['avg_weighted_probability'])
        print(f"ROC AUC: {auc:.4f}")
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = score(grouped['label'], grouped['prediction'], average='binary')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Calculate Matthews correlation coefficient
        mcc = matthews_corrcoef(grouped['label'], grouped['prediction'])
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(grouped['label'], grouped['prediction'])
        print("Confusion Matrix:")
        print(cm)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    # Count label distribution
    positive_count = (grouped['label'] == 1).sum()
    total_count = len(grouped)
    print(f"Positive cases: {positive_count}/{total_count} ({positive_count/total_count:.2%})")
    
    return grouped


def aggregate_general_results(input_dir, output_file, fold_pattern="fold_*_results.csv"):
    """
    Aggregates image results across multiple folds and computes averages.
    Preserves all individual image entries (not grouped by patient).
    
    Args:
        input_dir: Directory containing the fold CSV files
        output_file: Path to save the aggregated results
        fold_pattern: Pattern to match fold CSV files
    """
    # Get absolute path if not already
    input_dir = os.path.abspath(input_dir)
    
    # Get all matching CSV files
    csv_files = glob.glob(os.path.join(input_dir, fold_pattern))
    
    if not csv_files:
        print(f"No CSV files found matching pattern '{fold_pattern}' in '{input_dir}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to aggregate")
    
    # Read all fold data
    fold_dfs = {}
    for file in csv_files:
        fold_name = os.path.basename(file).split('_')[1]
        fold_dfs[fold_name] = pd.read_csv(file)
        print(f"Fold {fold_name}: {len(fold_dfs[fold_name])} rows")
    
    # Use the first fold as reference for ordering
    reference_fold = list(fold_dfs.keys())[0]
    reference_data = fold_dfs[reference_fold]
    
    # Process each row in the reference fold
    result_rows = []
    processed_tracking = set()  # Track which entries we've processed
    
    print(f"Processing {len(reference_data)} rows from reference fold {reference_fold}")
    
    for idx, row in reference_data.iterrows():
        patient_id = row['patient_id']
        
        # Create a unique identifier for this entry
        # We'll use position in the fold to distinguish multiple entries from same patient
        entry_position = sum(1 for r in result_rows if r['patient_id'] == patient_id)
        entry_id = f"{patient_id}_{entry_position}"
        
        if entry_id in processed_tracking:
            continue  # Skip if we've already processed this entry
        
        processed_tracking.add(entry_id)
        
        # Collect corresponding rows from all folds
        matching_rows = [row]
        
        # Look for matching rows in other folds
        for fold_name, fold_df in fold_dfs.items():
            if fold_name == reference_fold:
                continue  # Skip reference fold
            
            # Find corresponding row in this fold
            # Match by patient_id and relative position
            patient_rows = fold_df[fold_df['patient_id'] == patient_id]
            if len(patient_rows) > entry_position:
                # Get the row at the same position as in reference fold
                matching_rows.append(patient_rows.iloc[entry_position])
        
        # Calculate averages across folds
        avg_probability = sum(r['probability'] for r in matching_rows) / len(matching_rows)
        avg_uncertainty = sum(r['uncertainty'] for r in matching_rows) / len(matching_rows)
        
        # Create a result row
        result_rows.append({
            'patient_id': patient_id,
            'label': row['label'],
            'avg_probability': avg_probability,
            'avg_uncertainty': avg_uncertainty,
            'prediction': 1 if avg_probability > 0.5 else 0
        })
    
    # Create a DataFrame from the results
    result_df = pd.DataFrame(result_rows)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}, with {len(result_df)} unique images")
    
    # Calculate and print some statistics
    accuracy = (result_df['prediction'] == result_df['label']).mean()
    print(f"Overall accuracy: {accuracy:.4f}")
    
    try:
        # Calculate ROC AUC
        auc = roc_auc_score(result_df['label'], result_df['avg_probability'])
        print(f"ROC AUC: {auc:.4f}")
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = score(result_df['label'], result_df['prediction'], average='binary')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Calculate Matthews correlation coefficient
        mcc = matthews_corrcoef(result_df['label'], result_df['prediction'])
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(result_df['label'], result_df['prediction'])
        print("Confusion Matrix:")
        print(cm)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    # Count label distribution
    positive_count = (result_df['label'] == 1).sum()
    total_count = len(result_df)
    print(f"Positive cases: {positive_count}/{total_count} ({positive_count/total_count:.2%})")
    
    return result_df


def plot_combined_roc_curves(level='image', use_log_scale=True):
    """
    Plot ROC curves for different models using pre-aggregated results files.
    
    Args:
        level: 'image' or 'patient' to specify the analysis level
        use_log_scale: Whether to use logarithmic scale for the axes
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    
    plt.figure(figsize=(7, 4))
    
    # Define models and their result file paths based on level
    base_path = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/"
    
    # Updated color scheme
    colors = ['#ffec0d', '#e21f26', '#0568af', '#002554', '#32baf2']
    
    if level == 'image':
        models = {
            #'Tabular': {'result_file': f'{base_path}aggregated_general_results_tab.csv', 'color': colors[0]},
            'Vision': {'result_file': f'{base_path}aggregated_general_results_img.csv', 'color': colors[1]},
            'Intra': {'result_file': f'{base_path}aggregated_general_results_intra.csv', 'color': colors[2]},
            'Inter': {'result_file': f'{base_path}aggregated_general_results_inter.csv', 'color': colors[3]},
            'Late': {'result_file': f'{base_path}aggregated_general_results_late.csv', 'color': colors[4]}
        }
        prob_column = 'avg_probability'
    else:  # patient level
        models = {
            'Tabular': {'result_file': f'{base_path}aggregated_patient_results_tab.csv', 'color': colors[0]},
            'Vision': {'result_file': f'{base_path}aggregated_patient_results_img.csv', 'color': colors[1]},
            'Intra': {'result_file': f'{base_path}aggregated_patient_results_intra.csv', 'color': colors[2]},
            'Inter': {'result_file': f'{base_path}aggregated_patient_results_inter.csv', 'color': colors[3]},
            'Late': {'result_file': f'{base_path}aggregated_patient_results_late.csv', 'color': colors[4]}
        }
        prob_column = 'avg_weighted_probability'
    
    # Plot ROC curve for each model
    for model_name, config in models.items():
        result_path = config['result_file']
        
        if not os.path.exists(result_path):
            print(f"Result file not found for {model_name}: {result_path}")
            continue
        
        # Load aggregated results
        try:
            df = pd.read_csv(result_path)
            
            # Extract labels and probabilities
            y_true = df['label'].values
            y_prob = df[prob_column].values
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=config['color'], linestyle='-',
                     lw=2, label=f"{model_name} (AUC: {roc_auc:.3f})")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Add diagonal reference line and configure scales with updated color
    if use_log_scale and level == 'image':
        # Create diagonal line in log-log space
        log_points = np.logspace(-4, 0, 100)
        plt.plot(log_points, log_points, color='#cfb571', linestyle=':', label='Random classifier')
        
        # Set log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-3, 1.05)
        plt.ylim(1e-1, 1.05)
        plt.legend(loc='lower left', fontsize=8)
        
    elif use_log_scale and level == 'patient':
        log_points = np.logspace(-4, 0, 100)
        plt.plot(log_points, log_points, color='#cfb571', linestyle=':', label='Random classifier')
        
        # Set log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-3, 1.0)
        plt.ylim(9e-1, 1.003)
        plt.legend(loc='best', fontsize=8, bbox_to_anchor=(0.95, 0.35))
        
    else:
        # Regular linear scale diagonal
        plt.plot([0, 1], [0, 1], color='#cfb571', linestyle=':', label='Random classifier')
        plt.xlim(-0.01, 1.02)
        plt.ylim(0, 1.02)
        plt.legend(loc='lower right', fontsize=8)
    
    # Add labels and grid
    plt.xlabel('False Positive Rate', fontsize=8)
    plt.ylabel('True Positive Rate', fontsize=8)
    plt.grid(alpha=0.3)
    
    # Apply a clean style
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.show()

def print_metrics_summary(level='image'):
    """
    Calculate and print metrics (accuracy, AUC, F1-Score, MCC) with standard deviations
    for each model using pre-aggregated results and original fold files.
    
    Args:
        level: 'image' or 'patient' to specify the analysis level
    """
    import pandas as pd
    import os
    import glob
    import numpy as np
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    base_path = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/"
    
    # Define models and their paths based on level
    if level == 'image':
        models = [
            ('Tabular', f'{base_path}tab', f'{base_path}aggregated_general_results_tab.csv', 'fold_*_test_results.csv', 'probability'),
            ('Vision', f'{base_path}img', f'{base_path}aggregated_general_results_img.csv', 'fold_*_test_results.csv', 'probability'),
            ('Intra', f'{base_path}intra', f'{base_path}aggregated_general_results_intra.csv', 'fold_*_test_results.csv', 'probability'),
            ('Inter', f'{base_path}inter', f'{base_path}aggregated_general_results_inter.csv', 'fold_*_test_results.csv', 'probability'),
            ('Late', f'{base_path}late', f'{base_path}aggregated_general_results_late.csv', 'fold_*_test_results.csv', 'probability')
        ]
        agg_prob_column = 'avg_probability'
    else:  # patient level
        models = [
            ('Tabular', f'{base_path}tab', f'{base_path}aggregated_patient_results_tab.csv', 'fold_*_patient_results.csv', 'weighted_probability'),
            ('Vision', f'{base_path}img', f'{base_path}aggregated_patient_results_img.csv', 'fold_*_patient_results_img.csv', 'weighted_probability'),
            ('Intra', f'{base_path}intra', f'{base_path}aggregated_patient_results_intra.csv', 'fold_*_patient_results.csv', 'weighted_probability'),
            ('Inter', f'{base_path}inter', f'{base_path}aggregated_patient_results_inter.csv', 'fold_*_patient_results.csv', 'weighted_probability'),
            ('Late', f'{base_path}late', f'{base_path}aggregated_patient_results_late.csv', 'fold_*_patient_results.csv', 'weighted_probability')
        ]
        agg_prob_column = 'avg_weighted_probability'
    
    print(f"\n--- {level.upper()} LEVEL METRICS SUMMARY ---\n")
    
    # Print model-specific metrics with standard deviations
    print("Model metrics with standard deviations:")
    print("-" * 100)
    print(f"{'Model':<10} | {'Accuracy':<18} | {'AUC':<18} | {'F1-Score':<18} | {'MCC':<18}")
    print(f"{'':10} | {'Mean':>8} {'±SD':>9} | {'Mean':>8} {'±SD':>9} | {'Mean':>8} {'±SD':>9} | {'Mean':>8} {'±SD':>9}")
    print("-" * 100)
    
    for model_name, fold_dir, agg_file, fold_pattern, prob_col in models:
        try:
            # 1. First get aggregated results for the mean metrics
            if not os.path.exists(agg_file):
                print(f"{model_name:<10} | Aggregated file not found: {agg_file}")
                continue
                
            # Load aggregated results
            agg_df = pd.read_csv(agg_file)
            
            # Extract true labels, probabilities, and predictions
            y_true = agg_df['label'].values
            
            # Handle different column names in different files
            if agg_prob_column in agg_df.columns:
                y_prob = agg_df[agg_prob_column].values
            elif 'probability' in agg_df.columns and level == 'image':
                y_prob = agg_df['probability'].values
            elif 'weighted_probability' in agg_df.columns and level == 'patient':
                y_prob = agg_df['weighted_probability'].values
            else:
                print(f"{model_name:<10} | Error: Cannot find probability column in aggregated file")
                continue
            
            # Get predictions (either from file or calculate)
            if 'prediction' in agg_df.columns:
                y_pred = agg_df['prediction'].values
            else:
                y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics from aggregated data
            agg_accuracy = accuracy_score(y_true, y_pred)
            agg_auc = roc_auc_score(y_true, y_prob)
            agg_precision, agg_recall, agg_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            agg_mcc = matthews_corrcoef(y_true, y_pred)
            
            # 2. Now calculate metrics for each fold to get standard deviations
            fold_metrics = {
                'accuracy': [],
                'auc': [],
                'f1': [],
                'mcc': []
            }
            
            # Get fold files
            fold_files = glob.glob(os.path.join(fold_dir, fold_pattern))
            
            if not fold_files:
                print(f"{model_name:<10} | {agg_accuracy:.4f} ±N/A    | {agg_auc:.4f} ±N/A    | "
                      f"{agg_f1:.4f} ±N/A    | {agg_mcc:.4f} ±N/A")
                continue
            
            # Process each fold file
            for fold_file in fold_files:
                try:
                    fold_df = pd.read_csv(fold_file)
                    
                    # Skip if empty
                    if len(fold_df) == 0:
                        continue
                    
                    fold_y_true = fold_df['label'].values
                    fold_y_prob = fold_df[prob_col].values
                    fold_y_pred = (fold_y_prob > 0.5).astype(int)
                    
                    # Calculate metrics for this fold
                    fold_metrics['accuracy'].append(accuracy_score(fold_y_true, fold_y_pred))
                    fold_metrics['auc'].append(roc_auc_score(fold_y_true, fold_y_prob))
                    _, _, fold_f1, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average='binary', zero_division=0)
                    fold_metrics['f1'].append(fold_f1)
                    fold_metrics['mcc'].append(matthews_corrcoef(fold_y_true, fold_y_pred))
                except Exception as e:
                    print(f"Error processing fold file {fold_file}: {e}")
            
            # Calculate standard deviations
            if fold_metrics['accuracy']:  # Check if we have any fold data
                acc_std = np.std(fold_metrics['accuracy'])
                auc_std = np.std(fold_metrics['auc'])
                f1_std = np.std(fold_metrics['f1'])
                mcc_std = np.std(fold_metrics['mcc'])
                
                # Print results with mean from aggregated and std from folds
                print(f"{model_name:<10} | {agg_accuracy:.4f} ±{acc_std:.4f} | {agg_auc:.4f} ±{auc_std:.4f} | "
                      f"{agg_f1:.4f} ±{f1_std:.4f} | {agg_mcc:.4f} ±{mcc_std:.4f}")
            else:
                # No fold data, print without std
                print(f"{model_name:<10} | {agg_accuracy:.4f} ±N/A    | {agg_auc:.4f} ±N/A    | "
                      f"{agg_f1:.4f} ±N/A    | {agg_mcc:.4f} ±N/A")
            
        except Exception as e:
            print(f"{model_name:<10} | Error: {e}")
    
    print("-" * 100)

# Run the function for both levels

# Add these imports at the top of your script
from scipy.stats import wilcoxon
from MLstatkit.stats import Delong_test

def compare_patient_uncertainties():
    """
    Compare model uncertainties using Wilcoxon signed-rank test on patient-level data.
    Adapted to work with the aggregated patient results files in the current workflow.
    """
    # Base path to results
    base_path = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/"
    
    # Define result files for each model
    models = {
        'Tabular': f'{base_path}aggregated_patient_results_tab.csv',
        'Vision': f'{base_path}aggregated_patient_results_img.csv',
        'Intra': f'{base_path}aggregated_patient_results_intra.csv',
        'Inter': f'{base_path}aggregated_patient_results_inter.csv',
        'Late': f'{base_path}aggregated_patient_results_late.csv'
    }
    
    # Load uncertainty data for each model
    uncertainties = {}
    for model_name, file_path in models.items():
        try:
            df = pd.read_csv(file_path)
            # Select the uncertainty column (adjust if your column name is different)
            if 'avg_mean_uncertainty' in df.columns:
                uncertainties[model_name] = df['avg_mean_uncertainty'].values
            elif 'avg_patient_uncertainty' in df.columns:
                uncertainties[model_name] = df['avg_patient_uncertainty'].values
            else:
                print(f"Warning: Uncertainty column not found in {model_name} results")
                continue
            
            print(f"Loaded {len(uncertainties[model_name])} patient uncertainties for {model_name}")
        except Exception as e:
            print(f"Error loading {model_name} uncertainties: {e}")
    
    # Prepare results dataframes
    model_names = list(uncertainties.keys())
    p_values = pd.DataFrame(index=model_names, columns=model_names)
    statistics = pd.DataFrame(index=model_names, columns=model_names)
    
    # Display uncertainty statistics for each model
    print("\n===== Uncertainty Statistics by Model =====")
    for name, values in uncertainties.items():
        print(f"{name}: Mean = {np.mean(values):.6f}, Median = {np.median(values):.6f}, Std = {np.std(values):.6f}")
    
    # Perform Wilcoxon tests for all pairs of models
    for i, mod1 in enumerate(model_names):
        for j, mod2 in enumerate(model_names):
            if i == j:  # Same model
                p_values.loc[mod1, mod2] = 1.0
                statistics.loc[mod1, mod2] = 0.0
                continue
                
            # Perform Wilcoxon signed-rank test
            try:
                #stat, p = wilcoxon(uncertainties[mod1], uncertainties[mod2], alternative='two-sided')
                stat, p = ranksums(uncertainties[mod1], uncertainties[mod2])
                p_values.loc[mod1, mod2] = p
                statistics.loc[mod1, mod2] = stat
            except Exception as e:
                print(f"Error in Wilcoxon test for {mod1} vs {mod2}: {e}")
                p_values.loc[mod1, mod2] = np.nan
                statistics.loc[mod1, mod2] = np.nan
    
    # Print results
    print("\n===== Wilcoxon Signed-Rank Test P-values =====")
    print(p_values.round(6))
    print("\n===== Wilcoxon Signed-Rank Test Statistics =====")
    print(statistics.round(2))
    
    # Calculate Bonferroni correction
    n_comparisons = len(model_names) * (len(model_names) - 1) / 2
    alpha_corrected = 0.05 / n_comparisons
    
    # Summarize significant differences after correction
    print(f"\n===== Significant Differences (Bonferroni-corrected α = {alpha_corrected:.6f}) =====")
    significant_count = 0
    for i, mod1 in enumerate(model_names):
        for j, mod2 in enumerate(model_names):
            if i < j:  # Check each pair only once
                if pd.isna(p_values.loc[mod1, mod2]):
                    continue
                    
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

def compare_patient_aucs_with_delong():
    """
    Compare model AUC scores using DeLong test on patient-level data.
    Adapted to work with the aggregated patient results files in the current workflow.
    """
    # Base path to results
    base_path = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/"
    
    # Define result files for each model
    models = {
        'Tabular': f'{base_path}aggregated_patient_results_tab.csv',
        'Vision': f'{base_path}aggregated_patient_results_img.csv',
        'Intra': f'{base_path}aggregated_patient_results_intra.csv',
        'Inter': f'{base_path}aggregated_patient_results_inter.csv',
        'Late': f'{base_path}aggregated_patient_results_late.csv'
    }
    
    # Load predictions and true labels for each model
    model_data = {}
    patient_ids_by_model = {}
    
    for model_name, file_path in models.items():
        try:
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {model_name}: {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            # Extract patient IDs, labels and probabilities
            if 'patient_id' in df.columns:
                patient_ids = df['patient_id'].values
                patient_ids_by_model[model_name] = set(patient_ids)
            else:
                print(f"Warning: patient_id column not found in {model_name} results")
                continue
                
            # Extract labels
            y_true = df['label'].values
            
            # Extract probabilities (adjust column name if different)
            if 'avg_weighted_probability' in df.columns:
                y_prob = df['avg_weighted_probability'].values
            else:
                print(f"Warning: Probability column not found in {model_name} results")
                continue
            
            # Verify data validity
            if len(y_true) == 0 or len(y_prob) == 0:
                print(f"Warning: Empty data for {model_name}")
                continue
                
            if len(y_true) != len(y_prob):
                print(f"Warning: Mismatched lengths for {model_name}: {len(y_true)} vs {len(y_prob)}")
                continue
                
            # Store data with patient IDs for alignment
            model_data[model_name] = {
                'patient_ids': patient_ids,
                'true': y_true,
                'prob': y_prob
            }
            
            # Calculate and print individual AUC
            auc = roc_auc_score(y_true, y_prob)
            print(f"{model_name} AUC: {auc:.4f} (n={len(y_true)})")
            
        except Exception as e:
            print(f"Error loading {model_name} data: {e}")
    
    # Check if we have loaded data for at least 2 models
    if len(model_data) < 2:
        print("Not enough models with valid data for comparison")
        return pd.DataFrame(), pd.DataFrame()
        
    # Check if patient IDs are consistent across models
    # This is crucial for paired statistical tests
    all_models = list(model_data.keys())
    for i in range(len(all_models)):
        for j in range(i+1, len(all_models)):
            model1 = all_models[i]
            model2 = all_models[j]
            common_patients = patient_ids_by_model[model1].intersection(patient_ids_by_model[model2])
            total_patients = len(patient_ids_by_model[model1].union(patient_ids_by_model[model2]))
            
            print(f"Models {model1} and {model2}: {len(common_patients)}/{total_patients} common patients")
            
            if len(common_patients) < total_patients:
                print(f"Warning: Patient mismatch between {model1} and {model2}")
    
    # Prepare result dataframes
    model_names = list(model_data.keys())
    p_values = pd.DataFrame(index=model_names, columns=model_names)
    z_scores = pd.DataFrame(index=model_names, columns=model_names)
    
    # Perform DeLong test for all pairs of models
    for i, mod1 in enumerate(model_names):
        for j, mod2 in enumerate(model_names):
            if i == j:  # Same model
                p_values.loc[mod1, mod2] = 1.0
                z_scores.loc[mod1, mod2] = 0.0
                continue
            
            # Get the predictions for both models
            try:
                # We need to ensure the same patients are used in both models and in the same order
                # Get common patient IDs
                common_patient_ids = set(model_data[mod1]['patient_ids']).intersection(
                    set(model_data[mod2]['patient_ids']))
                
                if len(common_patient_ids) == 0:
                    print(f"No common patients between {mod1} and {mod2}")
                    p_values.loc[mod1, mod2] = np.nan
                    z_scores.loc[mod1, mod2] = np.nan
                    continue
                
                # Create masks for common patients in both models
                mask1 = np.isin(model_data[mod1]['patient_ids'], list(common_patient_ids))
                mask2 = np.isin(model_data[mod2]['patient_ids'], list(common_patient_ids))
                
                # Get aligned data
                patient_ids1 = model_data[mod1]['patient_ids'][mask1]
                patient_ids2 = model_data[mod2]['patient_ids'][mask2]
                y_true1 = model_data[mod1]['true'][mask1]
                y_true2 = model_data[mod2]['true'][mask2]
                y_prob1 = model_data[mod1]['prob'][mask1]
                y_prob2 = model_data[mod2]['prob'][mask2]
                
                # Create a DataFrame for each model to ensure proper alignment
                df1 = pd.DataFrame({'patient_id': patient_ids1, 'true': y_true1, 'prob': y_prob1})
                df2 = pd.DataFrame({'patient_id': patient_ids2, 'true': y_true2, 'prob': y_prob2})
                
                # Merge on patient_id to ensure alignment
                merged = pd.merge(df1, df2, on='patient_id', suffixes=('_1', '_2'))
                
                # Verify labels are the same
                if not np.array_equal(merged['true_1'], merged['true_2']):
                    print(f"Warning: Label mismatch between {mod1} and {mod2} for some patients")
                    # We'll use the labels from the first model
                
                # Get aligned data for DeLong test
                aligned_true = merged['true_1'].values
                aligned_prob1 = merged['prob_1'].values
                aligned_prob2 = merged['prob_2'].values
                
                print(f"Comparing {mod1} vs {mod2} on {len(aligned_true)} common patients")
                
                # Perform DeLong test
                
                try:
                    z_score, p_value = Delong_test(aligned_true, aligned_prob1, aligned_prob2)
                    if np.isnan(p_value):
                        print(f"Warning: NaN p-value for {mod1} vs {mod2}. AUCs might be identical.")
                        p_value = 1.0
                        z_score = 0.0
                except Exception as e:
                    print(f"Error running DeLong test for {mod1} vs {mod2}: {e}")
                    p_value = np.nan
                    z_score = np.nan
                #z_score, p_value = Delong_test(aligned_true, aligned_prob1, aligned_prob2)
                
                # Store results
                p_values.loc[mod1, mod2] = p_value
                z_scores.loc[mod1, mod2] = z_score
                
            except Exception as e:
                print(f"Error in DeLong test for {mod1} vs {mod2}: {e}")
                p_values.loc[mod1, mod2] = np.nan
                z_scores.loc[mod1, mod2] = np.nan
    
    # Print results
    print("\n===== DeLong Test P-values =====")
    print(p_values.round(6))
    print("\n===== DeLong Test Z-scores =====")
    print(z_scores.round(4))
    
    # Calculate Bonferroni correction
    n_comparisons = len(model_names) * (len(model_names) - 1) / 2
    alpha_corrected = 0.05 / n_comparisons
    
    # Summarize significant differences after correction
    print(f"\n===== Significant Differences (Bonferroni-corrected α = {alpha_corrected:.6f}) =====")
    significant_count = 0
    for i, mod1 in enumerate(model_names):
        for j, mod2 in enumerate(model_names):
            if i < j:  # Check each pair only once
                if pd.isna(p_values.loc[mod1, mod2]):
                    continue
                    
                p = p_values.loc[mod1, mod2]
                if p < alpha_corrected:
                    # Calculate AUCs on aligned data for a fair comparison
                    common_patients = set(model_data[mod1]['patient_ids']).intersection(
                        set(model_data[mod2]['patient_ids']))
                    
                    mask1 = np.isin(model_data[mod1]['patient_ids'], list(common_patients))
                    mask2 = np.isin(model_data[mod2]['patient_ids'], list(common_patients))
                    
                    auc1 = roc_auc_score(model_data[mod1]['true'][mask1], model_data[mod1]['prob'][mask1])
                    auc2 = roc_auc_score(model_data[mod2]['true'][mask2], model_data[mod2]['prob'][mask2])
                    
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
if __name__ == "__main__":
    """results_path = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/aggregated_general_results_late.csv"
    df = pd.read_csv(results_path)

    # Set style and figure size
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 4))

    # Create histogram with specified color
    sns.histplot(df["avg_uncertainty"], bins=20, color='#32baf2', edgecolor='black', alpha=0.9)

    # Customize axis labels
    plt.xlabel("Average Uncertainty")
    plt.ylabel("Frequency")

    # Remove spines
    sns.despine(left=True, bottom=True)"""
    
    #input_dir = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/late"
    #output_file = "/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/results/aggregated_general_results_late.csv"
    
    # Run the aggregation for image results
    #aggregate_general_results(input_dir, output_file, fold_pattern="fold_*_test_results.csv")
    
    plot_combined_roc_curves('image',use_log_scale=False)
    plot_combined_roc_curves('image',use_log_scale=True)
    #print_metrics_summary('patient')    
    
     # Perform statistical tests
    #print("\n===== Statistical Tests for Patient-Level Uncertainties =====")
    #uncertainty_p_vals, uncertainty_stats = compare_patient_uncertainties()
    
    #print("\n===== Statistical Tests for Patient-Level AUC Scores =====")
    #auc_p_vals, auc_z_scores = compare_patient_aucs_with_delong()
    
# %%