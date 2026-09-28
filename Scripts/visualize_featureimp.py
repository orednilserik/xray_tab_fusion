# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support as score
from tqdm import tqdm
import os
import torch.nn as nn
import timm
from vision_train import *
from sklearn.metrics import roc_auc_score, matthews_corrcoef 
def visualize_feature_importances(importances_dict, title_prefix="2", 
                                 figsize=(16, 6)):

    df = pd.DataFrame({
        'Feature': list(importances_dict.keys()),
        'Importance': list(importances_dict.values())
    }).sort_values('Importance', ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Distribution of all feature importances
    sns.histplot(df['Importance'], kde=True, ax=ax1, color='#32baf2')
    ax1.set_xlabel('Importance Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Feature Importance Distribution for fold {title_prefix}')
    
    # Plot 2: Top 10 features
    top10 = df.head(10)
    sns.barplot(y='Importance', x='Feature', data=top10, ax=ax2, color='#002554')
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    ax2.set_title(f'Highest Feature Importances for fold {title_prefix}')
    
    plt.tight_layout()
    
    plt.show()
    return fig

def vision_uncertainties(fold_no='fold_5'):
    # Load the data
    vis_img_data = torch.load(f'fusion_pipeline/{fold_no}/test_results_img.pth', weights_only=False)
    vis_img_df = vis_img_data['test_results']

    vis_pat_df = pd.read_csv(f'image_checkpoints/{fold_no}/patient_results_mc_weighted.csv')

    print(vis_img_df.head())
    print(vis_pat_df.head())
    print(len(vis_img_df))
    print(len(vis_pat_df))

    # compare patient_id in vis_pat_df with unique patient_id in vis_img_df
    print("Unique patient_id in vis_pat_df:", len(vis_pat_df['patient_id']))
    print("Unique patient_id in vis_img_df:", len(vis_img_df['patient_id'].unique()))
    print("Unique patient_id in vis_pat_df:", vis_pat_df['patient_id'])
    print("Unique patient_id in vis_img_df:", vis_img_df['patient_id'].unique())

    set_pat = set(vis_pat_df['patient_id'])
    set_img = set(vis_img_df['patient_id'].unique())
    if set_pat == set_img:
        print("The sets of patient_ids are equal.")
    else:
        print("The sets of patient_ids are not equal.")

    # create histogram of vis_img_df['uncertainty']
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_img_df['uncertainty'], kde=True, color='#32baf2')
    plt.xlabel('Uncertainty Value (Inverse standard deviation from MC Dropout)')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution Image-level')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_pat_df['mean_uncertainty'], kde=True, color='#32baf2')
    plt.xlabel('Uncertainty Value (Inverse standard deviation from MC Dropout)')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution Patient-level')
    plt.show()

def tabular_uncertainties(fold_no='fold_5'):
    # Load the data
    vis_img_data = torch.load(f'fusion_pipeline/{fold_no}/test_results_tab.pth', weights_only=False)
    vis_img_df = vis_img_data['test_results']

    

    print(vis_img_df.head())

    print(len(vis_img_df))

    # compare patient_id in vis_pat_df with unique patient_id in vis_img_df

    print("Unique patient_id in vis_img_df:", len(vis_img_df['patient_id'].unique()))

    print("Unique patient_id in vis_img_df:", vis_img_df['patient_id'].unique())

    # create histogram of vis_img_df['uncertainty']
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_img_df['uncertainty'], kde=True, color='#32baf2')
    plt.xlabel('Uncertainty Value (Inverse standard deviation from MC Dropout)')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution Patient-level')
    plt.show()
    

def fusion_uncertainties(fusion_dir, fold_no='fold_5'):
    
    vis_img_df = pd.read_csv(f'{fusion_dir}/{fold_no}/test_results.csv')
    vis_pat_df = pd.read_csv(f'{fusion_dir}/{fold_no}/patient_results_mc_weighted.csv')

    print(vis_img_df.head())
    print(vis_pat_df.head())
    print(len(vis_img_df))
    print(len(vis_pat_df))

    # compare patient_id in vis_pat_df with unique patient_id in vis_img_df
    print("Unique patient_id in vis_pat_df:", len(vis_pat_df['patient_id']))
    print("Unique patient_id in vis_img_df:", len(vis_img_df['patient_id'].unique()))
    print("Unique patient_id in vis_pat_df:", vis_pat_df['patient_id'])
    print("Unique patient_id in vis_img_df:", vis_img_df['patient_id'].unique())

    set_pat = set(vis_pat_df['patient_id'])
    set_img = set(vis_img_df['patient_id'].unique())
    if set_pat == set_img:
        print("The sets of patient_ids are equal.")
    else:
        print("The sets of patient_ids are not equal.")

    # create histogram of vis_img_df['uncertainty']
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_img_df['uncertainty'], kde=True, color='#32baf2')
    plt.xlabel('Uncertainty Value (Inverse standard deviation from MC Dropout)')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution Image-level')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(vis_pat_df['mean_uncertainty'], kde=True, color='#32baf2')
    plt.xlabel('Uncertainty Value (Inverse standard deviation from MC Dropout)')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution Patient-level')
    plt.show()





def aggregate_patient_results(test_df, inv_weight = True, fold_no=None):
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
    patient_results.to_csv(f'fusion_pipeline/{fold_no}/patient_results_mc_weighted.csv', index=False)
    
    # confusion matrix for patient-level predictions
    cm = confusion_matrix(patient_results['label'], patient_results['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NFF", "AFF"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues')
    plt.title('Vision Patient-Level Confusion Matrix (MC Dropout)')
    plt.savefig(f'fusion_pipeline/{fold_no}/patient_level_confusion_matrix_mc.png')
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
    plt.savefig(f'fusion_pipeline/{fold_no}/patient_level_uncertainty.png')
    plt.show()
    
    return patient_results




# Example usage
if __name__ == "__main__":
    """
    checkpoint_img = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/fusion_pipeline/fold_5/'
    vis_img_data = torch.load(os.path.join(checkpoint_img, 'test_results_img.pth'), weights_only=False)
    tab_img_data = torch.load(os.path.join(checkpoint_img, 'test_results_tab.pth'), weights_only=False)
    
    vis_res_img = vis_img_data['test_results']
    
    
    #aggregate_patient_results(vis_res_img, inv_weight=True, fold_no='fold_5')
    
    #
    """
    #vision_uncertainties('fold_1')
    #tabular_uncertainties('fold_1')
    fusion_uncertainties('inter_fusion_checkpoints', fold_no='fold_5')
    
    """
    feature_importances = {
        'bisfo_d': 0.1034,
        'age': 0.0281,
        'gica_d': 0.0260,
        'hrt': 0.0240,
        'galla_d': 0.0232,
        'osteopor_d': 0.0226,
        'hipsl_d': 0.0225,
        'psyke_d': 0.0224,
        'cvd_d': 0.0216,
        'proton': 0.0216,
        'atherosc_d': 0.0212,
        'endokrin_d': 0.0209,
        'index': 0.0207,
        'infekt_d': 0.0207,
        'sex': 0.0207,
        'gica2_d': 0.0206,
        'pancreas_d': 0.0204,
        'ihd_d': 0.0204,
        'opfraktsl_d': 0.0204,
        'muskskel_d': 0.0204,
        'ami_d': 0.0203,
        'kort': 0.0201,
        'kotasl_d': 0.0201,
        'diabetes_d': 0.0200,
        'resp_d': 0.0194,
        'cvlisch_d': 0.0189,
        'cvl_d': 0.0188,
        'mage_d': 0.0186,
        'kardiomy_d': 0.0186,
        'fraktsl_d': 0.0185,
        'blod_d': 0.0185,
        'hyperton_d': 0.0184,
        'nerv_d': 0.0183,
        'hud_d': 0.0183,
        'antidepr': 0.0180,
        'ra_d': 0.0177,
        'trochsl_d': 0.0177,
        'malign_d': 0.0175,
        'serm': 0.0175,
        'lipid_d': 0.0174,
        'urin_d': 0.0173,
        'antiep': 0.0172,
        'cervsl_d': 0.0171,
        'hjartsv_d': 0.0167,
        'cvlhem_d': 0.0166
    }
    
    # Visualize the feature importances
    visualize_feature_importances(feature_importances, 
                                 title_prefix="1")
    """
# %%