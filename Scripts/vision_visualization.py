# %%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import tqdm
import matplotlib.colors as mcolors
gradient_colors = ['#ffec0d','#e21f26', '#0568af', '#cfb571'] # sköld gul röd blå, guld
discrete_colors = ['#002554', '#32baf2']

dif_gradient1 = mcolors.LinearSegmentedColormap.from_list('dif_gradient', gradient_colors)
dif_gradient2 = mcolors.LinearSegmentedColormap.from_list('dif_gradient', discrete_colors)

# %%
source_path = "/local/data1/mikmo937/THESIS/Data/images"

# initialize counts and lists
patient_counts = Counter()
label_counts = Counter()
image_sizes = []
unique_patients = {'AFF': set(), 'CONTROL': set()}

# image file name
pattern = re.compile(r'patient_(\d+)_([A-Za-z]+)_class_1_modality_(?:CR|DX|XA)_scanoccasion_0_imagenumber_(\d+).png')

# iterate through all img files
for file_name in os.listdir(source_path):
    if file_name.endswith(".png"):
        match = pattern.match(file_name)
        if match:
            patient_id, label, img_nr = match.groups()
            patient_id, img_nr = int(patient_id), int(img_nr)
            
            # count patients and labels
            patient_counts[patient_id] += 1
            label_counts[label] += 1
            
            # track unique patients for each label
            if label in unique_patients:
                unique_patients[label].add(patient_id)


            # image dimensions
            img_path = os.path.join(source_path, file_name)
            img = plt.imread(img_path)
            height, width = img.shape[:2]
            image_sizes.append((width, height))

# image sizes as NumPy array
image_sizes = np.array(image_sizes)

# variable names from 'CONTROL' to NFF
label_counts['NFF'] = label_counts.pop('CONTROL')
unique_patients['NFF'] = unique_patients.pop('CONTROL')

# calculate number of unique patients for each label
unique_patient_counts = {label: len(patients) for label, patients in unique_patients.items()}
# %%
############################ PLOT 1: IMAGES PER FRACTURE TYPE
plt.figure(figsize=(6, 6))
sns.barplot(x=list(label_counts.keys()), 
            y=list(label_counts.values()),
            palette=[discrete_colors[0], discrete_colors[1]])
plt.xlabel("Fracture type")
plt.ylabel("Amount of images")
plt.savefig("plot1_patients_per_label.pdf")
plt.show()

print(label_counts)
# %%
############################ PLOT 2: IMAGES PER PATIENT
image_label_counts = Counter(patient_counts.values())
plt.figure(figsize=(6, 6))
sns.barplot(x=list(image_label_counts.keys()), 
            y=list(image_label_counts.values()), 
            palette=[discrete_colors[1]])
plt.xlabel("Number of images")
plt.ylabel("Amount of patients")
plt.savefig("plot2_images_per_patient.pdf")
plt.show()

# %%
############################ PLOT 3: PATIENTS PER FRACTURE TYPE
plt.figure(figsize=(6, 6))
sns.barplot(x=list(unique_patient_counts.keys()), 
            y=list(unique_patient_counts.values()), 
            palette=[discrete_colors[0], discrete_colors[1]])
plt.xlabel("Fracture type")
plt.ylabel("Amount of patients patients")
plt.savefig("plot3_patients_per_label.pdf")
plt.show()


# %%
############################ PLOT 1,2,3: IMAGES PER FRACTURE TYPE, IMAGES PER PATIENT, PATIENTS PER FRACTURE TYPE
plt.figure(figsize=(18, 6))

# Plot 1: Images per fracture type
plt.subplot(1, 3, 1)
sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()),
            palette=[discrete_colors[0], discrete_colors[1]])
plt.xlabel("Fracture type")
plt.ylabel("Amount of images")
plt.title("Images per fracture type")

# Plot 2: Images per patient
plt.subplot(1, 3, 2)
image_label_counts = Counter(patient_counts.values())
sns.barplot(x=list(image_label_counts.keys()), y=list(image_label_counts.values()),
            palette=[discrete_colors[0], discrete_colors[1]])
plt.xlabel("Number of images")
plt.ylabel("Amount of patients")
plt.title("Images per patient")

# Plot 3: Patients per fracture type
plt.subplot(1, 3, 3)
sns.barplot(x=list(unique_patient_counts.keys()), y=list(unique_patient_counts.values()),
            palette=[discrete_colors[0], discrete_colors[1]])
plt.xlabel("Fracture type")
plt.ylabel("Amount of patients")
plt.title("Patients diagnosed with each fracture type")

plt.tight_layout()
plt.savefig("plot_1_2_3.pdf", bbox_inches='tight')
plt.show()


# %%
############################ PLOT 4: IMAGE SIZE DISTRIBUTION
plt.figure(figsize=(8, 6))

# Create a custom Blues colormap that excludes the lightest shades
from matplotlib.colors import LinearSegmentedColormap
blues = plt.cm.Blues(np.linspace(0.3, 1, 256))  # Start from 0.3 instead of 0 to skip lightest colors
custom_blues = LinearSegmentedColormap.from_list('custom_blues', blues)

# Create a 2D histogram with explicit coloring and colorbar
kdeplot = sns.kdeplot(
    x=image_sizes[:, 0],
    y=image_sizes[:, 1],
    cmap=dif_gradient2,  # Color map with distinct colors (alternatives: plasma, inferno, jet)
    fill=True,
    thresh=0.05,
    bw_adjust=0.5,
    levels=30,    
    alpha=1.0,      
    cbar=True,       
    legend=False,
    linewidths=0.5  
)

median_width = np.median(image_sizes[:, 0])
median_height = np.median(image_sizes[:, 1])
kdeplot.axvline(median_width, color='red', linestyle='--')
kdeplot.axhline(median_height, color='red', linestyle='--')

# Customize the colorbar
cbar = kdeplot.collections[-1].colorbar
cbar.set_label("Density")

plt.xlabel("Image width")
plt.ylabel("Image height")
plt.savefig("plot4_image_size_distribution.pdf")
plt.show()



# %%
############################ PLOT 4: IMAGE SIZE HISTOGRAM
plt.figure(figsize=(8, 6))


# Create a 2D histogram (heatmap) with explicit coloring and colorbar
hist, xedges, yedges = np.histogram2d(image_sizes[:, 0], image_sizes[:, 1], bins=10)

# Plot the heatmap
plt.imshow(hist.T, 
           origin='lower', 
           cmap=dif_gradient2, 
           aspect='auto', 
           extent=[xedges[0], 
                   xedges[-1], 
                   yedges[0], 
                   yedges[-1]])
plt.colorbar(label='Count of Images')

median_width = np.median(image_sizes[:, 0])
median_height = np.median(image_sizes[:, 1])
plt.axvline(median_width, color='red', linestyle='--')
plt.axhline(median_height, color='red', linestyle='--')

plt.xlabel("Image width")
plt.ylabel("Image height")
plt.savefig("plot4_image_size_histogram.pdf")
plt.show()

min_width = np.min(image_sizes[:, 0])
max_width = np.max(image_sizes[:, 0])
median_width = np.median(image_sizes[:, 0])
min_height = np.min(image_sizes[:, 1])
max_height = np.max(image_sizes[:, 1])
median_height = np.median(image_sizes[:, 1])

print(f"Width: min={min_width}, max={max_width}, median={median_width}")
print(f"Height: min={min_height}, max={max_height}, median={median_height}")
# %%