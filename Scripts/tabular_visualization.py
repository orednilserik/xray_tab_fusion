# %%
from tabular_preprocess import visualization_dataset as data
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
gradient_colors = ['#ffec0d','#e21f26', '#0568af', '#cfb571'] # sköld gul röd blå, guld
discrete_colors = ['#002554', '#32baf2']

dif_gradient1 = mcolors.LinearSegmentedColormap.from_list('dif_gradient', gradient_colors)
dif_gradient2 = mcolors.LinearSegmentedColormap.from_list('dif_gradient', discrete_colors)
###################  AGE DISTRIBUTION
# %%
plt.figure(figsize=(6, 6))
sns.histplot(data['age'], 
             kde = True, 
             color=discrete_colors[1])

median_age = data['age'].median()
plt.axvline(median_age, 
            color='#e21f26', 
            linestyle='--')
plt.text(median_age, 
         plt.ylim()[1]*-0.05, 
         int(median_age), 
         color='#e21f26', 
         ha='center')
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("plot5_age_distribution.pdf")
plt.show()

################### SEX DISTRIBUTION
# %%
gender_map = {1: 'Male', 2: 'Female'}
genders = data['sex'].value_counts().rename(index=gender_map)

plt.figure(figsize=(6, 6))
sns.barplot(x=genders.index, 
            y=genders.values, 
            palette=[discrete_colors[0], 
                     discrete_colors[1]], 
            hue = genders.index, 
            legend = False)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.savefig("plot6_sex_distribution.pdf")
plt.show()

################### OSTEOPOROSIS ANALYSIS
# %%
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12,6))
#plt.figure(figsize=(6, 6))
osteopor_filtered = data[data['osteopor_d'] != 0]['osteopor_d']  # filter out 0s
sns.histplot(osteopor_filtered, 
             bins=20, 
             kde=True, 
             ax=ax1, 
             color=discrete_colors[1])

# Update x-axis labels for histogram
positions = ax1.get_xticks()
labels = [str(int(x)) for x in positions]
if '0' in labels:
    idx = labels.index('0')
    labels[idx] = '1'
ax1.set_xticks(positions)
ax1.set_xticklabels(labels)

# Set limits for histogram
min_val = osteopor_filtered.min()
max_val = osteopor_filtered.max()
ax1.set_xlim(min_val * -170, max_val * 1.05)
ax1.set_xlabel("Days")
ax1.set_ylabel("Patient count")

osteoporosis_map = {0: 'No', 1: 'Yes'}
osteopor_counts = data['osteopor'].value_counts().rename(index=osteoporosis_map)
print(osteopor_counts)
sns.barplot(x=osteopor_counts.index, y=osteopor_counts.values, 
            palette=[discrete_colors[0], discrete_colors[1]], ax=ax2)
ax2.set_xlabel("Diagnosis present or not")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig("plot7_osteoporosis_distribution.pdf")
plt.show()


################### BISPHOSPHONATE TREATMENT ANALYSIS
# %%
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12,6))
bisphosphonate_filtered = data[data['bisfo_d'] != 0]['bisfo_d']  # filter out 0s
sns.histplot(bisphosphonate_filtered, 
             bins=20, 
             kde=True, 
             ax=ax1, 
             color=discrete_colors[1])

# Update x-axis labels for histogram
positions = ax1.get_xticks()
labels = [str(int(x)) for x in positions]
if '0' in labels:
    idx = labels.index('0')
    labels[idx] = '1'
ax1.set_xticks(positions)
ax1.set_xticklabels(labels)

# Set limits for histogram
min_val = bisphosphonate_filtered.min()
max_val = bisphosphonate_filtered.max()
ax1.set_xlim(min_val * -1.01, max_val * 1.05)
ax1.set_xlabel("Days")
ax1.set_ylabel("Patient count")

bisphosphonate_map = {0: 'No', 1: 'Yes'}
bisphosphonate_counts = data['bisfo'].value_counts().rename(index=bisphosphonate_map)
print(bisphosphonate_counts)
sns.barplot(x=bisphosphonate_counts.index, y=bisphosphonate_counts.values, 
            palette=[discrete_colors[0],discrete_colors[1]], ax=ax2)
ax2.set_xlabel("Treatment present or not")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig("plot9_bisphosponate_distribution.pdf")
plt.show()
# %%

# %%
################### AGE AND SEX COMBINED DISTRIBUTION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Age distribution plot
sns.histplot(data['age'], 
             kde=True, 
             ax=ax1, 
             color=discrete_colors[1])
median_age = data['age'].median()
ax1.axvline(median_age, 
            color='#e21f26', 
            linestyle='--')
ax1.text(median_age, ax1.get_ylim()[1]*-0.050, f"{int(median_age)}", color='red', ha='center')
ax1.set_title("Age distribution")
ax1.set_xlabel("Age")
ax1.set_ylabel("Count")

# Sex distribution plot
gender_map = {1: 'Male', 2: 'Female'}
genders = data['sex'].value_counts().rename(index=gender_map)
sns.barplot(x=genders.index, 
            y=genders.values, 
            palette=[discrete_colors[0], 
                     discrete_colors[1]], 
            ax=ax2)
ax2.set_title("Gender distribution")
ax2.set_xlabel("Genders")
ax2.set_ylabel("Count")

plt.suptitle("Age and Gender Distribution", fontsize=14)
plt.tight_layout()
plt.savefig("plot5_6_age_gender_distribution.pdf")
plt.show()




# %%
################### CORRELATION BETWEEN OSTEOPOROSIS AND BISPHOSPHONATE TREATMENT DURATION
import scipy.stats as stats

# Filter out zeros from both columns to analyze only treated patients
treated_patients = data[(data['osteopor_d'] != 0) & (data['bisfo_d'] != 0)]

# Calculate correlation measures
pearson_corr, pearson_p = stats.pearsonr(treated_patients['osteopor_d'], 
                                         treated_patients['bisfo_d'])
# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Create the joint KDE plot
sns.kdeplot(x='osteopor_d', 
            y='bisfo_d', 
            data=treated_patients, 
            fill=True, 
            cmap=dif_gradient2, 
            levels=10, 
            alpha=0.9, 
            ax=ax)

# Add scatter plot of actual points
ax.scatter(treated_patients['osteopor_d'], 
           treated_patients['bisfo_d'], 
           c='#cfb571', 
           alpha=1, 
           s=20, 
           edgecolor='none')

"""
# Add a diagonal line for reference
max_val = max(treated_patients['osteopor_d'].max(), 
              treated_patients['bisfo_d'].max())
ax.plot([0, max_val], 
        [0, max_val], 
        'r--', 
        alpha=0.7)
"""
ax.set_xlim(left=-50)
ax.set_ylim(bottom=-50)
# Add correlation information as text
x_ticks = ax.get_xticks()
if 0 in x_ticks:
    new_x_labels = [str(int(1)) if x == 0 else str(int(x)) for x in x_ticks]
    ax.set_xticklabels(new_x_labels)

# Modify y-axis ticks
y_ticks = ax.get_yticks()
if 0 in y_ticks:
    new_y_labels = [str(int(1)) if y == 0 else str(int(y)) for y in y_ticks]
    ax.set_yticklabels(new_y_labels)
correlation_text = (
    f"Pearson correlation: {pearson_corr:.3f}"
)
ax.text(0.05, 0.95, 
        correlation_text, 
        transform=ax.transAxes, 
        bbox=dict(boxstyle='round', 
                  facecolor='white', 
                  alpha=0.8),
        verticalalignment='top')

# Set labels and title
ax.set_xlabel("Days since osteoporosis diagnosis")
ax.set_ylabel("Days since bisphosphonate treatment started")

plt.tight_layout()
plt.savefig("plot10_treatment_correlation.pdf")
plt.show()


# %%
################### BISPHOSPHONATE TREATMENT DURATION BY STRESS FRACTURE
# Include all patients, including those with bisfo_d = 0
#bisfo_treated = data.copy()
bisfo_treated = data[data['bisfo_d'] > 0].copy()
bisfo_treated['stressfx_Y'] = bisfo_treated['stressfx_Y'].map({0: 'NFF', 1: 'AFF'})

# Print the stress fracture value counts to verify labeling
stress_fx_counts = bisfo_treated['stressfx_Y'].value_counts()
print("Stress Fracture Counts:")
print(stress_fx_counts)

# Create the violin plot
plt.figure(figsize=(8, 6))
# Store the violinplot object
vp = sns.violinplot(x='stressfx_Y', 
                   y='bisfo_d', 
                   data=bisfo_treated, 
                   palette=[discrete_colors[0], discrete_colors[1]],
                   inner='quartile',
                   cut=0)

# Change the colors of the inner lines (quartiles)
for l in vp.lines:
    l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('#e21f26')  # Red color from your gradient_colors
    l.set_alpha(0.8)
    
# For the horizontal lines (median)
for l in vp.lines[1::3]:
    l.set_linewidth(2)
    l.set_color('#e21f26')  # Different color for median if desired

# Add individual data points
sns.stripplot(x='stressfx_Y', 
              y='bisfo_d', 
              data=bisfo_treated, 
              color='#cfb571',
              alpha=1, 
              jitter=True,
              size=4)

# Set labels and title
plt.xlabel("Fracture type")
plt.ylabel("Days since bisphosphonate treatment started")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("plot11_bisphosphonate_by_stress_fracture.pdf")
plt.show()



# %%
################### BISPHOSPHONATE TREATMENT DURATION BY STRESS FRACTURE
# Include all patients, including those with bisfo_d = 0
#osteo_diag = data.copy()
osteo_diag = data[data['osteopor_d'] > 0].copy()
osteo_diag['sex'] = osteo_diag['sex'].map({1: 'Male', 2: 'Female'})

# Create the violin plot
plt.figure(figsize=(8, 6))
# Store the violinplot object
vp = sns.violinplot(x='sex', 
                   y='osteopor_d', 
                   data=osteo_diag, 
                   palette=[discrete_colors[0], discrete_colors[1]],
                   inner='quartile',
                   cut=0)

# Change the colors of the inner lines (quartiles)
for l in vp.lines:
    l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('#e21f26')  # Red color from your gradient_colors
    l.set_alpha(0.8)
    
# For the horizontal lines (median)
for l in vp.lines[1::3]:
    l.set_linewidth(2)
    l.set_color('#e21f26')  # Different color for median if desired

# Add individual data points
sns.stripplot(x='sex', 
              y='osteopor_d', 
              data=osteo_diag, 
              color='#cfb571',
              alpha=1, 
              jitter=True,
              size=4)

# Set labels and title
plt.xlabel("Sex")
plt.ylabel("Days since osteoporosis diagnosis")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("plot12_osteoporosis_gender.pdf")
plt.show()

# %%
