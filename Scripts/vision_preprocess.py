# %%
import os
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.io import write_png, read_image, ImageReadMode, decode_image
import torch.multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from extract_patients import split_dataset, source_path
from scipy import ndimage as ndi

# utilize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define path to images
#source_path = "/local/data1/mikmo937/THESIS/Data/images_one_per_patient"
#source_path = "/local/data1/mikmo937/THESIS/Data/images"

# augmentations from previous master's thesis
def previous_augmentations():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=0.3 * 360),  # convert radians to degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # random translation and zoom
        #transforms.ColorJitter(contrast=0.3)  # random contrast adjustment
    ])

class ImgData(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data, self.patient_dict = self._load_data() # return both list and dict
        self.class_weights = self._calculate_class_weights() # calculate class weights
         
         
    def _load_data(self):
        data = [] # initialize data list
        patient_dict = {} # patient dict
        
        for file_name in tqdm(os.listdir(self.path), desc = "Loading images"): # iterate through images
            if file_name.endswith(".png"): # all images should end with .png
                # image and patient number from current iteration
                img_path = os.path.join(self.path, file_name)
                patient_number = int(file_name.split('_')[1])
                
                # load image as grayscale
                img = decode_image(img_path)
                img = self._preprocess_image(img, file_name) # pad, resize, normalize intensity
                
                label = 1 if "_AFF_" in file_name else 0 # label 1 if AFF and 0 if NFF
                
                # append list
                data.append((patient_number, img, label))
                
                # fill dict with unique images per patient
                if patient_number not in patient_dict:
                    patient_dict[patient_number] = {
                        "images": [img],
                        "label": label
                    }
                else:
                    patient_dict[patient_number]["images"].append(img)
                
        return data, patient_dict
                
    def _preprocess_image(self, img, file_name):
        # pad and resize
        height, width = img.shape[1], img.shape[2]
        #height_c, width_c = height // 2, width // 2
        pad_size = max(height, width)
        #pad_c = pad_size // 2
        pad_img = torch.zeros((1, pad_size, pad_size)) # create padded square img
        
        # calculate padding
        pad_height = (pad_size - height) // 2
        pad_width = (pad_size - width) // 2
    
        # place image in center of the padded img
        pad_img[:, pad_height:pad_height + height, pad_width:pad_width + width] = img

        # resize image to 224x224
        pad_img_resized = transforms.Resize((224, 224), antialias = True)(pad_img)
        
        
        # normalize intensities to [-1, 1]
        pad_img_normalized = transforms.Normalize(mean=[0.5], std=[0.5])(pad_img_resized)
       
        return pad_img_normalized
    
    def _calculate_class_weights(self):
        labels = [label for _, _, label in self.data] # get labels from the labels in data
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
        return torch.tensor([class_weights.max() / class_weights.min()], dtype=torch.float32, device=device)
  
    # used by Dataset class to determine size of dataset
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        patient_number, img, label = self.data[idx]
        if self.transform:
            img = self.transform(img) # apply transformations# retrieve idx sample from data
        # convert label to torch.long tensor and move to GPU
        return patient_number, img, torch.as_tensor(label, dtype=torch.long)

# subset indices based on patient IDs
train_patients, val_patients, test_patients = split_dataset()

class PatientSubset(Subset):
        def __getitem__(self, idx):
            data_idx = self.indices[idx]   # index into original dataset
            pat, img, label = self.dataset.data[data_idx]
            
            # apply any dataset-level transform if needed
            if self.dataset.transform:
                img = self.dataset.transform(img)
            
            return pat, img, torch.as_tensor(label, dtype=torch.long)

def create_subset(dataset, train_patients = train_patients, val_patients = val_patients, test_patients = test_patients):
    
    # create index lists for subsets
    train_indices = [i for i, (pat, _, _) in enumerate(dataset.data) if pat in train_patients]
    val_indices = [i for i, (pat, _, _) in enumerate(dataset.data) if pat in val_patients]
    test_indices = [i for i, (pat, _, _) in enumerate(dataset.data) if pat in test_patients]
    
    # use lists to subset dataset
    train_dataset = PatientSubset(dataset, train_indices)
    val_dataset = PatientSubset(dataset, val_indices)
    test_dataset = PatientSubset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

# used to iterate over the respective datasets
def create_dataloaders(train, val, test, batch_size = 32, cv = False):
    if cv:
        cpu_threads = 0
    else:
        cpu_threads = os.cpu_count() // 2 # number of threads to use
    # shuffle = True to shuffle data at the beginning of each epoch
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True,
                              num_workers = cpu_threads, pin_memory = True, persistent_workers=True)
    
    # shuffle is not necessary for val and test data
    val_loader = DataLoader(val, batch_size = batch_size, shuffle = False,
                            num_workers = cpu_threads, pin_memory = True, persistent_workers=True)
    
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False,
                            num_workers = cpu_threads, pin_memory = True, persistent_workers=True)
    
    return train_loader, val_loader, test_loader

# sanity checks by counting classes, looking for duplicates and plotting samples
def count_classes(dataset, original_data):
    class_counts = Counter()
    for idx in dataset.indices:
        label = original_data.data[idx][2]
        class_counts[int(label)] += 1
    return class_counts

def count_duplicates(lst):
    return len(lst) != len(set(lst))

def plot_samples(dataset, title, n_samples=5):
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    fig.suptitle(title, fontsize=16)
    
    for i in range(n_samples):
        img, label = dataset[i]
        img = img.squeeze().cpu().numpy()  # convert tensor to numpy array
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.show()

def plot_intensity(dataset, title):
    intensities = []
    for img, _ in dataset:
        intensities.extend(img.flatten().cpu().numpy())
    
    plt.figure(figsize=(10, 5))
    plt.hist(intensities, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()

def main():
    mp.set_start_method("spawn")
    
    transform = previous_augmentations()
    images = ImgData(source_path, transform=transform)
    
    train_data, val_data, test_data = create_subset(images)

    # length of datasets
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")
    print(f"Number of test samples: {len(test_data)}")
    print("\n")
    
    # amount of samples per class for each dataset   
    train_class_counts = count_classes(train_data, images)
    val_class_counts = count_classes(val_data, images)
    test_class_counts = count_classes(test_data, images)
    print(f"Training class distribution: {train_class_counts}")
    print(f"Validation class distribution: {val_class_counts}")
    print(f"Test class distribution: {test_class_counts}")
    print("\n")
    
    # check for patient duplicates among splits to ensure no data leak
    all_patients = train_patients + val_patients + test_patients
    duplicates_across = [x for x in all_patients if all_patients.count(x) > 1]

    print("Duplicates in training set:", count_duplicates(train_patients))
    print("Duplicates in validation set:", count_duplicates(val_patients))
    print("Duplicates in test set:", count_duplicates(test_patients))
    print("Duplicates across all sets:", set(duplicates_across), "(should return empty set)")
    print("\n")
    
    print("Class weights:", images.class_weights)
    print("\n")
    # plot samples from each dataset
    plot_samples(train_data, "Training Samples")
    plot_samples(val_data, "Validation Samples")
    plot_samples(test_data, "Test Samples")
    
    # plot intensity histograms
    #plot_intensity(train_data, "Training Data Intensity Distribution")
    #plot_intensity(val_data, "Validation Data Intensity Distribution")
    #plot_intensity(test_data, "Test Data Intensity Distribution")
    
    print('-' * 100)

if __name__ == "__main__":
    main()
# %%