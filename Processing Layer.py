import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import logging
from datetime import datetime
import glob
import editDFTest as testDF
from SpecDataHandler import SpecDataHandler

# Load sample info
sampleInfoSheet_df = pd.read_excel('sampleSpecData/sampleSheetData/SampleSheet2.xlsx', sheet_name=0).dropna()

# Find all text files in the folder
specFiles = glob.glob('sampleSpecData/Data - Absorption/*.txt')



# Create DataFrames from all text files
dfs = [
    pd.read_csv(file, delimiter='\t', header=None, names=["Wavelength", "Absorption"]).assign(Sample=f'Sample_{i}')
    for i, file in enumerate(specFiles)
]

# Combine DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Pivot to reshape the DataFrame
pivot_df = combined_df.pivot(index='Wavelength', columns='Sample', values='Absorption').reset_index()

# Define target wavelengths
target_wavelengths = [630.188, 710.104, 800.131, 905.029, 940.061, 1000.111]

# Filter based on target wavelengths
filtered_pivot_df = pivot_df[pivot_df['Wavelength'].isin(target_wavelengths)]

# Function to order columns numerically
def order_columns(df):
    ordered_columns = ['Wavelength'] + sorted(
        [col for col in df.columns if col.startswith('Sample_') or col.startswith('Material_') or col.startswith('STDMaterial_')],
        key=lambda x: int(x.split('_')[1])
    )
    return ordered_columns

# Reorder columns
ordered_filtered_pivot_df = filtered_pivot_df[order_columns(filtered_pivot_df)]

#testDF.output_val(ordered_filtered_pivot_df,"ordered_data")

def create_std_material_df(dataFrame):
    # Extract all sample columns (ignoring 'Wavelength')
    sample_columns = [col for col in dataFrame.columns if col != 'Wavelength']

    # Create a mapping to assign a material label to each sample, similar to create_averaged_material_df
    material_labels = []
    for i, sample in enumerate(sample_columns):
        material_number = (i // 3) + 1  # Group every 3 samples
        material_label = f'Material_{material_number}'
        material_labels.append(material_label)

    # Create a dictionary to rename sample columns by their material group
    material_mapping = dict(zip(sample_columns, material_labels))

    # Rename columns in the DataFrame based on the material mapping
    renamed_df = dataFrame.rename(columns=material_mapping)

    # Group columns by material name to calculate standard deviation for each group
    std_material_df = pd.DataFrame()
    std_material_df['Wavelength'] = dataFrame['Wavelength']

    # Iterate over unique material labels and calculate std deviation for each group
    for material_label in set(material_labels):
        material_columns = [col for col in renamed_df.columns if col == material_label]

        # Only calculate if we have more than one column per material (i.e., grouped samples)
        if len(material_columns) > 1:
            std_material_df[f'STD{material_label}'] = renamed_df[material_columns].std(axis=1)
    std_material_df = std_material_df[order_columns(std_material_df)]
    return std_material_df
# Function to create averaged DataFrame by material
def create_averaged_material_df(dataFrame):
    # Extract all sample columns (ignoring 'Wavelength')
    sample_columns = [col for col in dataFrame.columns if col != 'Wavelength']

    # Create a mapping to assign a material label to each sample
    material_labels = []
    for i, sample in enumerate(sample_columns):
        material_number = (i // 3) + 1  # Group every 3 samples
        material_label = f'Material_{material_number}'
        material_labels.append(material_label)

    # Create a dictionary to rename sample columns by their material group
    material_mapping = dict(zip(sample_columns, material_labels))

    # Rename columns in the DataFrame based on the material mapping
    renamed_df = dataFrame.rename(columns=material_mapping)

    # Group by 'Wavelength' and then average columns with the same material name
    averaged_material_df = renamed_df.groupby('Wavelength', axis=0).mean().reset_index()

    # Average the columns with the same material name by grouping them along axis=1
    averaged_material_df = averaged_material_df.groupby(averaged_material_df.columns, axis=1).mean()

    # Reorder columns numerically
    averaged_material_df = averaged_material_df[order_columns(averaged_material_df)]
    return averaged_material_df

grouped_material_pivot_df = create_averaged_material_df(ordered_filtered_pivot_df)
#create_std_df(ordered_filtered_pivot_df)
# Assuming grouped_material_pivot_df is your pivot DataFrame (it was created earlier in your script)
mat = SpecDataHandler(grouped_material_pivot_df) #material based grouping (averaged)
sam = SpecDataHandler(ordered_filtered_pivot_df) #sample based grouping (filtered)
std = create_std_material_df(ordered_filtered_pivot_df)


print("materials")
mat.print_stats()


print("samples")
sam.print_stats()


print("std")
print(std)

# Now you can call the dataset_absorbance method on this instance

# Print and save the DataFrame
#testDF.output_val(grouped_material_pivot_df,"grouped_data")

#print(sampleInfoSheet_df)



