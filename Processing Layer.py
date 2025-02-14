"""
This script processes spectroscopic data from text files and performs various analyses including:
- Loading and combining multiple spectroscopic data files
- Filtering data for specific wavelengths
- Calculating material averages and standard deviations
- Exporting processed data to Excel

The script handles both sample-level and material-level data processing.
"""

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
sampleInfoSheet_df = pd.read_excel('sampleSpecData/sampleSheetData/scattering_samples_02_13_25.xlsx', sheet_name=0).dropna()

# Find all text files in the folder
specFiles = glob.glob('sampleSpecData/Data - Absorption/*.txt')



# Create DataFrames from all text files
dfs = [
    pd.read_csv(file, delimiter='\t', header=0, skiprows=14,
                names=["Wavelength", "Absorption"]).assign(Sample=f'Sample_{i}')
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
    """
    Orders DataFrame columns numerically based on their suffix numbers.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns to be ordered
        
    Returns:
        list: Ordered list of column names with 'Wavelength' first, followed by
              numerically sorted Sample_, Material_, or STDMaterial_ columns
    """
    ordered_columns = ['Wavelength'] + sorted(
        [col for col in df.columns if col.startswith('Sample_') or col.startswith('Material_') or col.startswith('STDMaterial_')],
        key=lambda x: int(x.split('_')[1])
    )
    return ordered_columns

# Reorder columns
ordered_filtered_pivot_df = filtered_pivot_df[order_columns(filtered_pivot_df)]

#testDF.output_val(ordered_filtered_pivot_df,"ordered_data")

def create_std_material_df(absorbance_df):
    """
    Calculates standard deviation of absorbance values for each material group.
    
    Args:
        absorbance_df (pandas.DataFrame): DataFrame containing absorbance values
            with samples grouped by material (3 samples per material)
            
    Returns:
        pandas.DataFrame: DataFrame containing wavelengths and standard deviations
            for each material group
    """
    std_material_df = pd.DataFrame()
    std_material_df['Wavelength'] = absorbance_df.index.values
    
    sample_columns = [col for col in absorbance_df.columns if col.startswith('Sample_')][:]  
    for i in range(0, len(sample_columns), 3):
        group_samples = sample_columns[i:i+3]
        material_number = (i // 3) + 1
        group_std = absorbance_df[group_samples].std(axis=1)
        std_material_df[f'STDMaterial_{material_number}'] = group_std.values
    
    std_material_df = std_material_df.reset_index(drop=True)
    return std_material_df

# Function to create averaged DataFrame by material
def create_averaged_material_df(dataFrame):
    """
    Creates a DataFrame with averaged values for each material group.
    
    Args:
        dataFrame (pandas.DataFrame): Input DataFrame containing sample data
            where every 3 samples represent one material
            
    Returns:
        pandas.DataFrame: DataFrame with averaged values for each material group
    """
    sample_columns = [col for col in dataFrame.columns if col != 'Wavelength']
    material_labels = []
    for i, sample in enumerate(sample_columns):
        material_number = (i // 3) + 1
        material_label = f'Material_{material_number}'
        material_labels.append(material_label)

    material_mapping = dict(zip(sample_columns, material_labels))
    renamed_df = dataFrame.rename(columns=material_mapping)
    averaged_material_df = renamed_df.groupby('Wavelength', axis=0).mean().reset_index()
    averaged_material_df = averaged_material_df.groupby(averaged_material_df.columns, axis=1).mean()
    averaged_material_df = averaged_material_df[order_columns(averaged_material_df)]
    return averaged_material_df

def create_absorbance_df(MatSpecData):
    """
    Creates an absorbance DataFrame from spectroscopic data.
    
    Args:
        MatSpecData (SpecDataHandler): Object containing spectroscopic data
            
    Returns:
        pandas.DataFrame: DataFrame containing ordered absorbance values
    """
    absorbance_df = MatSpecData.dataset_absorbance()
    
    if 'Wavelength' in absorbance_df.columns:
        absorbance_df = absorbance_df[order_columns(absorbance_df)]
    else:
        ordered_columns = sorted(
            [col for col in absorbance_df.columns if col.startswith('Sample_') or col.startswith('Material_') or col.startswith('STDMaterial_')],
            key=lambda x: int(x.split('_')[1])
        )
        absorbance_df = absorbance_df[ordered_columns]

    return absorbance_df

grouped_material_pivot_df = create_averaged_material_df(ordered_filtered_pivot_df)
#create_std_df(ordered_filtered_pivot_df)
mat = SpecDataHandler(grouped_material_pivot_df)  # Material_1 will be reference
sam = SpecDataHandler(ordered_filtered_pivot_df)  # First sample will be reference
absorbance_df_sam = create_absorbance_df(sam)
std = create_std_material_df(absorbance_df_sam)
absorbance_df_mat = create_absorbance_df(mat)

print("absorbance sample")
print(absorbance_df_sam)

print("absorbance material")
print(absorbance_df_mat)

print("materials")
mat.print_stats()


print("samples")
sam.print_stats()


print("std")
print(std)

#Define the output Excel file path
output_file_path = 'combined_dataframes.xlsx'

# Create an Excel writer object and specify the file path
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    # Write each DataFrame to a specific sheet/tab
    mat.return_all().to_excel(writer, sheet_name='Materials', index=False)
    sam.return_all().to_excel(writer, sheet_name='Samples', index=False)
    std.to_excel(writer, sheet_name='Standard Deviations', index=False)
    absorbance_df_sam.to_excel(writer, sheet_name='Absorbance Sample', index=False)
    absorbance_df_mat.to_excel(writer, sheet_name='Absorbance Material', index=False)
