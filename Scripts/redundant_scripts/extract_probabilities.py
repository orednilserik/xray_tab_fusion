"""
FINAL MERGED PROBABILITIES
merged_train_df
merged_val_df
merged_test_df
"""

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

# READING AND LOADING DATA FROM UNIMODAL MODELS
# %%


tab_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'
img_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'
# read data and load dfs
tab_checkpoint = torch.load(tab_path + 'best_tabular_cv.pth', weights_only=False)
img_checkpoint = torch.load(img_path + 'best_vision_cv.pth', weights_only=False)
tab_test = torch.load(tab_path + 'test_results.pth', weights_only=False)
img_test = torch.load(img_path + 'test_results.pth', weights_only=False)

tab_train_df = tab_checkpoint['train_results']
tab_val_df = tab_checkpoint['val_results']
tab_test_df = tab_test['test_results']

img_train_df = img_checkpoint['train_results']
img_val_df = img_checkpoint['val_results']
img_test_df = img_test['test_results']
#img_test_df = pd.read_csv('/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/test_results.csv')

print(tab_train_df.head())
print(tab_val_df.head())
print(img_train_df.head())
print(img_val_df.head())
print(img_train_df.shape)
print(tab_train_df.shape)

print(set(tab_train_df['patient_id']))
print(set(img_train_df['patient_id']))

########## MERGING DATA
# %%
merged_train_df = img_train_df.merge(tab_train_df, on='patient_id',
                                     how = 'inner', suffixes=('_img', '_tab'))
merged_val_df = img_val_df.merge(tab_val_df, on='patient_id',
                                 how = 'inner', suffixes=('_img', '_tab'))
merged_test_df = img_test_df.merge(tab_test_df, on='patient_id',
                                   how = 'inner', suffixes=('_img', '_tab'))
# drop label_tab column
merged_test_df = merged_test_df.drop(columns=['label_tab'])
merged_train_df = merged_train_df.drop(columns=['label_tab'])
merged_val_df = merged_val_df.drop(columns=['label_tab'])
# rename label_img column
merged_train_df = merged_train_df.rename(columns={'label_img': 'label'})
merged_val_df = merged_val_df.rename(columns={'label_img': 'label'})
merged_test_df = merged_test_df.rename(columns={'label_img': 'label'})

merged_train_df = merged_train_df.sort_values(by='patient_id')
merged_val_df = merged_val_df.sort_values(by='patient_id')
merged_test_df = merged_test_df.sort_values(by='patient_id')
print(merged_train_df.head())
print(merged_val_df.head())
print(merged_test_df.head())
print(merged_train_df.shape)
print(merged_val_df.shape)
print(merged_test_df.shape)

print(merged_train_df.columns)
print(merged_val_df.columns)
print(merged_test_df.columns)

"""
FINAL MERGED PROBABILITIES
merged_train_df
merged_val_df
merged_test_df
"""