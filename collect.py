import os
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import resample_img
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Get the script's directory
script_dir = os.path.dirname(__file__)

# Load the downloaded Harvard-Oxford atlas (keep atlas paths absolute)
atlas_file_path = r"C:\spyder_files\versionOne\OASIS-1_dataset-master\HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
atlas_img = nib.load(atlas_file_path)

# Load a sample NIfTI image to check its shape and affine
sample_nifti_path = r"C:\spyder_files\versionOne\OASIS-1_dataset-master\sub-OASIS10001_ses-M00_T1w.nii.gz"
sample_nifti_img = nib.load(sample_nifti_path)

# Visualize the original and resampled Harvard-Oxford Atlas
plotting.plot_roi(atlas_img, title="Original Harvard-Oxford Atlas")
atlas_resampled = resample_img(
    atlas_img, target_affine=sample_nifti_img.affine, target_shape=sample_nifti_img.shape)
plotting.plot_roi(atlas_resampled, title="Resampled Harvard-Oxford Atlas")
plt.show()

# Get voxel size
voxel_size = np.abs(np.linalg.det(atlas_resampled.affine[:3, :3]))

# Set up paths relative to the script directory
raw_dir = os.path.join(script_dir, 'OASIS-1_dataset-master', 'raw')
tsv_file_path = os.path.join(
    script_dir, 'OASIS-1_dataset-master', 'tsv_files', 'lab_1', 'OASIS_BIDS.tsv')

# Dictionary to store patient data
patient_data = {}

# Load neuropsychological data
neuro_data = pd.read_csv(tsv_file_path, sep='\t')


def calculate_roi_volumes(nifti_img, atlas_data, voxel_size):
    data = nifti_img.get_fdata()
    roi_volumes = {}

    # Iterate through unique ROI labels in the atlas
    for roi_label in np.unique(atlas_data):
        if roi_label == 0:  # Skip the background label
            continue

        # Create a mask for the current ROI
        roi_mask = (atlas_data == roi_label)

        # Count the voxels in the ROI
        roi_voxel_count = np.sum(roi_mask & (data > 0))
        roi_volume = roi_voxel_count * voxel_size  # Convert voxel count to volume

        roi_volumes[roi_label] = roi_volume

    return roi_volumes


# Get the total number of files for progress tracking
total_files = len([f for f in os.listdir(raw_dir) if f.endswith('.nii.gz')])
print(f"Total NIfTI files to process: {total_files}")

# Process each NIfTI file
for index, file_name in enumerate(os.listdir(raw_dir)):
    if file_name.endswith('.nii.gz'):
        file_path = os.path.join(raw_dir, file_name)
        nifti_img = nib.load(file_path)

        # Resample the atlas to the NIfTI image's space
        resampled_atlas = resample_img(
            atlas_img, target_affine=nifti_img.affine, target_shape=nifti_img.shape)

        # Calculate ROI volumes
        roi_volumes = calculate_roi_volumes(
            nifti_img, resampled_atlas.get_fdata(), voxel_size)

        # Assuming the patient ID is part of the file name
        patient_id = file_name.split('_')[0]

        # Store the result in the patient_data dictionary
        patient_data[patient_id] = roi_volumes

        # Print progress every 10 files
        if (index + 1) % 10 == 0 or (index + 1) == total_files:
            print(f"Processed {index + 1}/{total_files}: {file_name}")

# Convert patient_data dictionary to a DataFrame
roi_volumes_df = pd.DataFrame.from_dict(patient_data, orient='index').fillna(0)
roi_volumes_df['total_volume'] = roi_volumes_df.sum(axis=1)
roi_volumes_df.index = roi_volumes_df.index.str.replace('sub-', '')

# Remove the 'sub-' prefix from the participant_id in the neuropsychological data to match the NIfTI participant_id format
neuro_data['participant_id'] = neuro_data['participant_id'].str.replace(
    'sub-', '')

# Merge neuropsychological data with the calculated ROI volumes based on the participant_id
combined_data = neuro_data.merge(
    roi_volumes_df, left_on='participant_id', right_index=True, how='inner')

print(combined_data)

# Combine with neuropsychological data
combined_data = neuro_data.merge(
    roi_volumes_df, left_on='participant_id', right_index=True, how='inner')
print(combined_data)

# Save the combined data to a new CSV file using relative path
combined_data.to_csv(os.path.join(
    script_dir, 'combined_patient_data.csv'), index=False)

# Select the last three columns
last_three_columns = combined_data.columns[-49:]

# Calculate standard deviation for each of the last three columns
std_devs = combined_data[last_three_columns].std()
top_std_columns = std_devs.nlargest(3).index.tolist()


# Updated list1 excluding 'laterality'
list1 = ['sex', 'education_level', 'age_bl', 'cdr',
         'diagnosis_bl', 'MMS', 'cdr_global', 'diagnosis']
list2 = top_std_columns
# Create a DataFrame for correlations between list1 and list2 columns
correlation_results = pd.DataFrame(index=list1, columns=list2)


# Sort the columns by standard deviation and get the top 3


# Add total_volume to list2

# Convert categorical variables in list1 to numeric
for col in list1:
    if col in combined_data.columns:
        combined_data[col], _ = pd.factorize(combined_data[col])

# Create a grid of scatter plots
num_rows = len(list1)
num_cols = len(list2)

# Adjust axes handling for single column
fig, axes = plt.subplots(num_rows, num_cols, figsize=(
    15, 4 * num_rows), sharex=True, sharey=True)

if num_cols == 1:  # If only one column, axes will be 1D
    axes = np.atleast_2d(axes).T  # Convert to 2D array to avoid index issues

# Plotting each combination
for i, col1 in enumerate(list1):
    if col1 in combined_data.columns:
        for j, col2 in enumerate(list2):
            if col2 in combined_data.columns:
                ax = axes[i, j] if num_cols > 1 else axes[i, 0]
                ax.scatter(combined_data[col1], combined_data[col2], alpha=0.6)
                ax.set_title(f'{col1} vs {col2}')
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)

                # Calculate and display the correlation coefficient
                clean_data = combined_data[[col1, col2]].dropna()
                if clean_data[col1].nunique() > 1 and clean_data[col2].nunique() > 1:
                    corr, _ = pearsonr(clean_data[col1], clean_data[col2])
                    ax.text(
                        0.05, 0.95, f'Corr: {corr:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')


# Proceed with the correlation and plotting as before
# Create a DataFrame for correlations between list1 and list2 columns
correlation_results = pd.DataFrame(index=list1, columns=list2)

# Calculate correlation between list1 and list2
for col1 in list1:
    for col2 in list2:
        clean_data = combined_data[[col1, col2]].dropna()
        if clean_data[col1].nunique() > 1 and clean_data[col2].nunique() > 1:
            corr, _ = pearsonr(clean_data[col1], clean_data[col2])
            correlation_results.at[col1, col2] = corr
        else:
            # Set to NaN if correlation can't be calculated
            correlation_results.at[col1, col2] = np.nan

# Plotting the correlation matrix between list1 and list2
plt.figure(figsize=(10, 8))
plt.title(
    'Correlation Matrix between List1 and Total Volume + Top 3 Columns by Std Dev')
sns.heatmap(correlation_results.astype(float), annot=True,
            cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Create a PDF file path
pdf_path = r"C:\spyder_files\combined_patient_data.pdf"  # Change the path as needed

# Create a new figure with a larger size for better readability
plt.figure(figsize=(16, 10))
plt.axis('tight')
plt.axis('off')

# Create a table from the DataFrame
table_data = combined_data.values
columns = combined_data.columns

# Create the table and add it to the figure
table = plt.table(cellText=table_data, colLabels=columns,
                  cellLoc='center', loc='center')

# Adjust font size and properties for better readability
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)  # Increase the scale for larger cells

# Manually adjust column widths for readability
for i in range(len(columns)):
    table.auto_set_column_width(i)

# Save the figure as a PDF
with PdfPages(pdf_path) as pdf:
    pdf.savefig()
    plt.close()

print(f"Combined data has been exported as a readable PDF table to {pdf_path}")


def print_tree(startpath, level=0, max_length=30, max_items=6):
    """Print the directory tree starting from the given path, with limits on item length and count."""
    # Print the current directory, truncating long names
    dir_name = os.path.basename(startpath)
    if len(dir_name) > max_length:
        dir_name = dir_name[:max_length] + '...'
    print('    ' * level + dir_name + '/')

    # Iterate over the contents of the directory
    try:
        items = sorted(os.listdir(startpath))  # Get all items in the directory
        for index, item in enumerate(items):
            if index >= max_items:
                # Indicate more items exist
                print('    ' * (level + 1) + '...')
                break

            item_path = os.path.join(startpath, item)
            # Truncate file names for display
            if len(item) > max_length:
                item = item[:max_length] + '...'

            if os.path.isdir(item_path):
                # Recursive call for directories
                print_tree(item_path, level + 1, max_length, max_items)
            else:
                print('    ' * (level + 1) + item)  # Print files
    except PermissionError:
        print('    ' * level + 'Permission Denied')


base_path = r'C:\spyder_files\versionOne'
print_tree(base_path)
