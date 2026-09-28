# %%
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
import random
import math
import torch.multiprocessing as mp
from collections import Counter
import numpy as np
#from extract_patients import split_dataset, load_patient_ID
from sklearn.utils.class_weight import compute_class_weight

source_paths = "/local/data1/mikmo937/THESIS/Data/images"

def load_patient_ID(source = source_paths):
    patient_set = set()  # patient set

    for file_name in os.listdir(source):  # iterate through images
        if file_name.endswith(".png"):  # all images should end with .png
            # image and patient number from current iteration
            patient_number = int(file_name.split('_')[1])
            patient_set.add(patient_number)  # add patient number to set

    return list(patient_set) # convert to list to make subscriptable

def split_dataset(train_frac = 0.8, val_frac = 0.1, seed = 181):
    patient_ids = load_patient_ID() # load patient IDs
    if seed is not None:
        random.seed(seed) # random seed for reproducibility
        random.shuffle(patient_ids) # random shuffle
    else:
        random.shuffle(patient_ids)
    
    # calculate amount per split
    total_patients = len(patient_ids)
    train_size = math.ceil(total_patients * train_frac)
    val_size = math.ceil(total_patients * val_frac)
    
    # split patient IDs
    train_patients = patient_ids[:train_size]
    val_patients = patient_ids[train_size:train_size + val_size]
    test_patients = patient_ids[train_size + val_size:]
    
    return train_patients, val_patients, test_patients

source_path = "/home/mikmo937/Desktop/DATA1/THESIS/Data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read data and reindex columns alphabetically
data = pd.read_csv(os.path.join(source_path, "registerdata_all.csv"))
#data = data.reindex(sorted(data.columns), axis = 1)

data.rename(columns = {'stressfx': 'stressfx_Y'}, inplace = True) # rename response variable for clarity

# drop non-numeric columns and rows with missing values (no missing values existed)
data_all = data[['age', 'antidepr', 'antiep', 'hrt', 'index', 'kort', 'proton', 'rlpnr', 'serm', 'sex', 'stressfx_Y']]
data_num = data.filter(like="_d", axis = 1).dropna()
data_num = pd.concat([data_all, data_num], axis = 1)

# drop patients not in image data
available_patients = load_patient_ID()
visualization_dataset = data[data['rlpnr'].isin(available_patients)] # using for visualization
dataset_for_use = data_num[data_num['rlpnr'].isin(available_patients)] # final dataset for subsequent preprocessing and subsetting
#print(visualization_dataset.head())
# %%
# indices to split dataset into training, validation and test sets
train_patients, val_patients, test_patients = split_dataset()

# map patient IDs to row indices in final dataset
train_indices = [i for i in range(len(dataset_for_use)) if dataset_for_use.iloc[i]['rlpnr'] in train_patients]
val_indices = [i for i in range(len(dataset_for_use)) if dataset_for_use.iloc[i]['rlpnr'] in val_patients]
test_indices = [i for i in range(len(dataset_for_use)) if dataset_for_use.iloc[i]['rlpnr'] in test_patients]

def preprocess_data(dataset = dataset_for_use, train_indices = train_indices, excluded_cols = ['rlpnr', 'stressfx_Y'],
                    val_indices = val_indices, test_indices = test_indices):
    
    categorical_cols = ['sex']
    
    # split features and labels
    continuous_cols = [col for col in dataset.columns if col not in excluded_cols+categorical_cols]
    df_continuous = dataset[continuous_cols]
    df_categorical = dataset[categorical_cols]
    df_excluded = dataset[excluded_cols] # patient ID and labels
    
    # standardize using only the training data
    scaler = StandardScaler()
    df_continuous = df_continuous.astype(float)
    df_continuous.iloc[train_indices] = scaler.fit_transform(df_continuous.iloc[train_indices])  # fit on train
    # only transform using train scaler to prevent data leakage
    df_continuous.iloc[val_indices] = scaler.transform(df_continuous.iloc[val_indices])  
    df_continuous.iloc[test_indices] = scaler.transform(df_continuous.iloc[test_indices])

    # combine standardized features with excluded columns
    final_df = pd.concat([df_excluded.reset_index(drop=True), 
                         df_categorical.reset_index(drop=True),
                         df_continuous.reset_index(drop=True)], axis=1)
    final_df['sex'].replace({2:0}, inplace=True)
    sex_col = final_df['sex']
    final_df.drop(labels=['sex'],axis=1, inplace=True)
    final_df.insert(0, 'sex', sex_col)
    return final_df

def create_subset(dataset, train_indices = train_indices, val_indices = val_indices, test_indices = test_indices):

    # separate features and labels
    features = dataset.drop(columns=['rlpnr', 'stressfx_Y']).values
    labels = dataset['stressfx_Y'].values
    patient_ids = dataset['rlpnr'].values

    # convert to tensors
    dataset_tensor = TensorDataset(torch.tensor(features, dtype=torch.float32),
                                   torch.tensor(labels, dtype=torch.float32),
                                   torch.tensor(patient_ids, dtype=torch.long))
                                    

    # create subsets
    train_dataset = Subset(dataset_tensor, train_indices)
    val_dataset = Subset(dataset_tensor, val_indices)
    test_dataset = Subset(dataset_tensor, test_indices)

    return train_dataset, val_dataset, test_dataset

# used to iterate over the respective datasets
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size = 32):
    cpu_threads = os.cpu_count() // 2 # number of threads to use
    
    # shuffle = True to shuffle data at the beginning of each epoch
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                              num_workers = cpu_threads, pin_memory = True, persistent_workers=True)
    
    # shuffle is not necessary for val and test data
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False,
                            num_workers = cpu_threads, pin_memory = True, persistent_workers=True)
    
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False,
                            num_workers = cpu_threads, pin_memory = True, persistent_workers=True)
    
    return train_loader, val_loader, test_loader

# class weights
def compute_class_weights(dataset):
    labels = np.array([label for _, label,_ in dataset]) # extract labels
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels) # compute class weights
    # compute positive ratio for bcewithlogitsloss
    pos_weight = torch.tensor(class_weights.max() / class_weights.min(), dtype=torch.float32, device=device)
    return pos_weight

def count_classes(dataset, original_data):
    class_counts = Counter()
    for idx in dataset.indices:
        label = original_data.iloc[idx]['stressfx_Y']
        class_counts[int(label)] += 1
    return class_counts

def count_duplicates(lst):
    return len(lst) != len(set(lst))

def main():
    mp.set_start_method("spawn")
    
    final_df = preprocess_data()
    train_data, val_data, test_data = create_subset(final_df)
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data)
    
    print('-' * 100)
    print('Columns:', "\n")
    print(final_df.columns, '\n')
    print(f'Shape: {final_df.shape}')
    print(f'Types:', "\n") 
    print(final_df.dtypes)
    print("\n")
    print(f'Head:', "\n")
    print(final_df.head())
    print("\n")


    # length of datasets
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of test samples: {len(test_data)}")
    print("\n")
    
    # amount of samples per class for each dataset   
    train_class_counts = count_classes(train_data, final_df)
    val_class_counts = count_classes(val_data, final_df)
    test_class_counts = count_classes(test_data, final_df)
    print(f"Training class distribution: {train_class_counts}")
    print(f"Validation class distribution: {val_class_counts}")
    print(f"Test class distribution: {test_class_counts}")
    print("\n")
    
    # check for patient duplicates among splits to ensure no data leak
    all_patients = train_indices + val_indices + test_indices
    duplicates_across = [x for x in all_patients if all_patients.count(x) > 1]

    print("Duplicates in training set:", count_duplicates(train_indices))
    print("Duplicates in validation set:", count_duplicates(val_indices))
    print("Duplicates in test set:", count_duplicates(test_indices))
    print("Duplicates across all sets:", set(duplicates_across), "(should return empty set)")
    print("\n")
    
    print(f"Class weights: {compute_class_weights(train_data)}")
    print('-' * 100)
    
    print(final_df['sex'].value_counts())
    
    print(final_df['index'].value_counts())
    print(final_df['hrt'].value_counts())
    print(final_df['serm'].value_counts())
    print(final_df['antiep'].value_counts())
    print(final_df['antidepr'].value_counts())
    print(final_df['kort'].value_counts())
    print(final_df['proton'].value_counts())
if __name__ == "__main__":
    main()
# %%