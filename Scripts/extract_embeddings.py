"""
Final merged embeddings
pair_train_emb = (pid, tab_emb, img_emb) x 3218
pair_val_emb = (pid, tab_emb, img_emb) x 406
pair_test_emb = (pid, tab_emb, img_emb) x 390
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from vision_train import VisionTransformer
from tabular_train import TabularTransformer
import os
from collections import defaultdict
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tabular_preprocess import compute_class_weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support as score
import json
from collections import namedtuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_embeddings(fold_no='fold_3'):
    #tab_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/tabular_checkpoints/'
    #img_path = '/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/image_checkpoints/'
    
    # read data and load dfs
    fusion_path='/home/mikmo937/Desktop/DATA1/THESIS/Github/Scripts/fusion_pipeline/'
    
    tab_checkpoint = torch.load(fusion_path + fold_no + '/best_tabular_model.pth', weights_only=False)
    img_checkpoint = torch.load(fusion_path + fold_no + '/best_vision_model.pth', weights_only=False)
    tab_test = torch.load(fusion_path + fold_no + '/test_results_tab.pth', weights_only=False)
    img_test = torch.load(fusion_path + fold_no + '/test_results_img.pth', weights_only=False)

    tab_train_df = tab_checkpoint['train_results']
    tab_val_df = tab_checkpoint['val_results']
    tab_test_df = tab_test['test_results']

    #img_train_df = img_checkpoint['train_results']
    #img_val_df = img_checkpoint['val_results']
    #img_test_df = img_test['test_results']

    tab_train_embeddings = tab_checkpoint['train_embeddings']
    tab_val_embeddings = tab_checkpoint['val_embeddings']
    tab_test_embeddings = tab_test['test_embeddings']

    img_train_embeddings = img_checkpoint['train_embeddings']
    img_val_embeddings = img_checkpoint['val_embeddings']
    img_test_embeddings = img_test['test_embeddings']

    print("Tab embedding key length:", len(tab_train_embeddings.keys()), len(tab_val_embeddings.keys()), len(tab_test_embeddings.keys()))
    print("Img embedding key length:", len(img_train_embeddings.keys()), len(img_val_embeddings.keys()),len(img_test_embeddings.keys()))


    unique_img_test_pids = {k.split('_')[0] for k in img_test_embeddings.keys()}
    print(f"Unique Img‑test patient count: {len(unique_img_test_pids)}")
    print(f"Tab‑test patient count:           {len(tab_test_embeddings.keys())}")

    # Get the first 10 patient IDs
    for pid in list(img_train_embeddings.keys())[:100]:  
        print(f"Patient ID: {pid}")

        # Get the first entry's keys (assuming there's at least one entry)
        if img_train_embeddings[pid]:
            entry_keys = img_train_embeddings[pid][0].keys()
            print("Value Names:", list(entry_keys))

        print("-" * 30)  # Separator for readability

    ################## MERGE ALL EMBEDDINGS!!!!
    merged_train_emb = {}
    merged_val_emb = {}
    merged_test_emb = {}

    for pid, img_emb_list in img_train_embeddings.items():
        if pid in tab_train_embeddings:
            tab_emb = tab_train_embeddings[pid]['embedding']
            merged_train_emb[pid] = {
                'tabular_embedding': tab_emb,
                'vision_embeddings': [info['embedding'] for info in img_emb_list]
            }

    for pid, img_emb_list in img_val_embeddings.items():
        if pid in tab_val_embeddings:
            tab_emb = tab_val_embeddings[pid]['embedding']
            merged_val_emb[pid] = {
                'tabular_embedding': tab_emb,
                'vision_embeddings': [info['embedding'] for info in img_emb_list]
            }      


    ################## MERGE TEST EMBEDDINGS BY BASE PID (drop the “_i” suffix) 
    # collapse vision MC runs
    img_by_pid = defaultdict(list)
    for entry_id, info_list in img_test_embeddings.items():
        base_pid = entry_id.split('_')[0]
        for info in info_list:
            img_by_pid[base_pid].append(info['embedding'])

    # collapse tabular MC runs
    tab_by_pid = defaultdict(list)
    for entry_id, entries in tab_test_embeddings.items():
        base_pid = entry_id.split('_')[0]
        if isinstance(entries, list):
            for info in entries:
                tab_by_pid[base_pid].append(info['embedding'])
        else:
            tab_by_pid[base_pid].append(entries['embedding'])

    # now build merged_test_emb keyed by real pid
    merged_test_emb = {}
    for pid, vision_embs in img_by_pid.items():
        tab_embs = tab_by_pid.get(pid, None)
        if not tab_embs:
            continue
        # average tabular MC embeddings
        tab_stack = torch.stack(tab_embs, dim=0)
        tab_mean  = tab_stack.mean(dim=0)
        merged_test_emb[pid] = {
            'tabular_embedding': tab_mean,
            'vision_embeddings': vision_embs
        }

    for pid, data in merged_train_emb.items():
        print(f"PID: {pid}")
        print(f"  - tabular_embedding: {type(data['tabular_embedding'])}")
        print(f"  - vision_embeddings: {type(data['vision_embeddings'])} (Length: {len(data['vision_embeddings'])})")
        if len(data['vision_embeddings']) > 0:
            print(f"    - First vision embedding: {type(data['vision_embeddings'][0])}")
        print("-" * 50)
# %%
    ################## SANITY CHECKS
    total_train_embeddings = sum(len(data['vision_embeddings']) for data in merged_train_emb.values())
    total_val_embeddings = sum(len(data['vision_embeddings']) for data in merged_val_emb.values())
    total_test_embeddings = sum(len(data['vision_embeddings']) for data in merged_test_emb.values())
    print(f"Training image embeddings: {total_train_embeddings}")
    print(f"Validation image embeddings: {total_val_embeddings}")
    print(f"Testing image embeddings: {total_test_embeddings}")
    print("\n")

    # 4014 imgs and 1073 tabular. CORRECT!
    # compare keys between the three merged embedding dictionaries
    train_keys = set(merged_train_emb.keys())
    val_keys = set(merged_val_emb.keys())
    test_keys = set(merged_test_emb.keys())

    print(f"Total unique keys in train set: {len(train_keys)}")
    print(f"Total unique keys in validation set: {len(val_keys)}")
    print(f"Total unique keys in test set: {len(test_keys)}")

    # check for overlaps between datasets
    train_val_overlap = train_keys.intersection(val_keys)
    train_test_overlap = train_keys.intersection(test_keys)
    val_test_overlap = val_keys.intersection(test_keys)

    print(f"\nOverlap between train and validation: {len(train_val_overlap)} keys")
    print(f"Overlap between train and test: {len(train_test_overlap)} keys")
    print(f"Overlap between validation and test: {len(val_test_overlap)} keys")

    # check for keys that appear in all three sets
    all_sets_overlap = train_keys.intersection(val_keys, test_keys)
    print(f"\nKeys that appear in all three sets: {len(all_sets_overlap)}")
    # %%
    ###################### PAIR ALL IMG EMBEDDINGS TO TAB EMBEDDINGS

    EmbeddingsInput = namedtuple('EmbeddingsInput', ['pid', 'tabular', 'vision', 'label'])

    pair_train_emb = []
    pair_val_emb = []
    pair_test_emb = []

    train_labels = {row['patient_id']: row['label'] for _, row in tab_train_df.iterrows()}
    val_labels = {row['patient_id']: row['label'] for _, row in tab_val_df.iterrows()}
    test_labels = {row['patient_id']: row['label'] for _, row in tab_test_df.iterrows()}

    for pid, data in merged_train_emb.items():
        tab_emb = data['tabular_embedding']
        label = int(train_labels.get(pid))
        for img_emb in data['vision_embeddings']:
            pair_train_emb.append(EmbeddingsInput(
                pid=pid,
                tabular=tab_emb.detach().cpu() if torch.is_tensor(tab_emb) else tab_emb,
                vision=img_emb.detach().cpu() if torch.is_tensor(img_emb) else img_emb,
                label=label
            ))
            
    for pid, data in merged_val_emb.items():
        tab_emb = data['tabular_embedding']
        label = int(val_labels.get(pid))
        for img_emb in data['vision_embeddings']:
            pair_val_emb.append(EmbeddingsInput(
                pid=pid,
                tabular=tab_emb.detach().cpu() if torch.is_tensor(tab_emb) else tab_emb,
                vision=img_emb.detach().cpu() if torch.is_tensor(img_emb) else img_emb,
                label=label
            ))
            
    # instead of “for pid, data in merged_test_emb.items(): …”
    for pid_str, data in merged_test_emb.items():
        # if your test_labels keys are ints, cast:
        pid = int(pid_str)  
        label = test_labels[pid]
        tab_emb = data['tabular_embedding']
        for vision_emb in data['vision_embeddings']:
            pair_test_emb.append( EmbeddingsInput(
                pid     = pid,
                tabular = tab_emb.detach().cpu() if torch.is_tensor(tab_emb) else tab_emb,
                vision  = vision_emb.detach().cpu() if torch.is_tensor(vision_emb) else vision_emb,
                label   = label
            ) )
        
    len(pair_train_emb), len(pair_val_emb), len(pair_test_emb)
    
    return pair_train_emb, pair_val_emb, pair_test_emb
