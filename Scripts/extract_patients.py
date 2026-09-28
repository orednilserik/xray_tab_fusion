import os
import random
import math


source_path = "/local/data1/mikmo937/THESIS/Data/images"
#source_path = "/local/data1/mikmo937/THESIS/Data/images_one_per_patient"


def load_patient_ID(source = source_path):
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

    



def main():  
    print(load_patient_ID())
    print("\n")
    print(f"Length of set is {len(load_patient_ID())}")
    print("-"*60)
if __name__ == "__main__":
    main()