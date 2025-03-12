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

from matplotlib.pyplot import scatter
from numpy.matlib import randn

import editDFTest as testDF
from SpecDataHandler import SpecDataHandler
import re

# Load sample info
sampleInfoSheet_df = pd.read_excel('sampleSpecData/sampleSheetData/absorber 02_13_25.xlsx', sheet_name=0).dropna()

# Find all text files in the folder
specFiles = glob.glob('sampleSpecData/Data - Absorption/*.txt')

# Load the Scattering info
scatteringInfoSheet_df = pd.read_excel('Scattering_Data/scattering_samples_02_13_25.xlsx', sheet_name='Data')
# Only drop rows where essential columns are missing
essential_columns = ['Material_ID', 'scatter_Volume_mg']
scatteringInfoSheet_df = scatteringInfoSheet_df.dropna(subset=essential_columns)
#Load the Scattering Data
scatteringDataSheet_df = pd.read_excel('Scattering_Data/Scattering_combined_dataframes.xlsx',sheet_name='Absorbance Material').dropna()


# Extract sample ID from filename pattern: ...__XX__XXXX.txt
def parse_sample_id(filename):
    match = re.search(r'__(\d+)__\d+(?:_void)?(?:\.txt)?$', filename)
    return int(match.group(1)) if match else None

# Modified file processing with ID parsing
dfs = [
    pd.read_csv(file, delimiter='\t', header=0, skiprows=14,
                names=["Wavelength", "Absorption"])
    .assign(Sample=f"Sample_{parse_sample_id(os.path.basename(file))}")
    for file in specFiles
]


# Combine DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Pivot to reshape the DataFrame
pivot_df = combined_df.pivot(index='Wavelength', columns='Sample', values='Absorption').reset_index()

# Define target wavelengths
target_wavelengths = [630.188, 710.104, 800.131, 905.029, 940.061] # removed wavelength of 1000.111

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

def filter_negative_val(absorbance_df):
    df=absorbance_df.mask(absorbance_df < 0)
    return df
ordered_filtered_pivot_df = filter_negative_val(ordered_filtered_pivot_df)


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
'''
def create_absorbance_df(MatSpecData):
    """
    Creates an absorbance DataFrame from spectroscopic data.
    
    Args:
        MatSpecData (SpecDataHandler): Object containing spectroscopic data
            
    Returns:
        pandas.DataFrame: DataFrame containing ordered absorbance values
    """
    absorbance_df = MatSpecData.dataset_absorbance()

    absorbance_df = [order_columns(absorbance_df)]


    return absorbance_df
'''

def create_absorbance_df(MatSpecData, ):
    absorbance_df = MatSpecData.dataset_absorbance()
    # Use the list of ordered columns to select those columns from the DataFrame
    absorbance_df = absorbance_df[order_columns(absorbance_df)]
    return absorbance_df


def create_scattering_map(dfInfoSheet):
    dfInfoSheet = dfInfoSheet[['Material_ID', 'scatter_Volume_mg']].copy()

    # Convert Material_ID to numeric first
    dfInfoSheet['Material_ID'] = pd.to_numeric(dfInfoSheet['Material_ID'], errors='coerce')

    scatter_map = {}

    for i, row in dfInfoSheet.iterrows():
        material_id = int(row['Material_ID'])  # Convert to integer
        scatter_val = row['scatter_Volume_mg']

        if pd.isna(scatter_val):  # Skip missing values
            continue

        if scatter_val not in scatter_map:
            scatter_map[scatter_val] = []

        # Add formatted material name instead of just the ID
        material_name = f"Material_{material_id}"

        # Only add if not already in the list
        if material_name not in scatter_map[scatter_val]:
            scatter_map[scatter_val].append(material_name)
    # Comment this part out to bring back the 0.0 scattering volume, it's non-zero b/c of the scattering that happens in the environment + noise I presume
    if np.float64(0.0) in scatter_map:
        scatter_map.pop(np.float64(0.0))
    return scatter_map


def average_absorbance_by_scatter(absorbance_df, scatter_map):
    """
    Averages absorbance values by scatter volume.

    Parameters:
        absorbance_df (pd.DataFrame): DataFrame containing 'Wavelength' and material columns
            with absorbance values (like 'Material_1', 'Material_2', etc.)
        scatter_map (dict): Dictionary mapping scatter volumes to lists of material names
            e.g., {0.0: ['Material_1', 'Material_2'], 12.8: ['Material_3', 'Material_4'], ...}

    Returns:
        pd.DataFrame: New DataFrame with 'Wavelength' and one column for each scatter volume,
            containing the average absorbance values for materials with that scatter volume
    """
    # Make sure the 'Wavelength' column exists
    if 'Wavelength' not in absorbance_df.columns:
        print("Error: 'Wavelength' column not found in absorbance data.")
        return None

    # Prepare a dictionary to build the new DataFrame
    new_data = {'Wavelength': absorbance_df['Wavelength']}

    # Loop over each scatter value and its corresponding list of material names
    for scatter_val, material_names in scatter_map.items():
        # Convert scatter value to string for column name
        scatter_key = str(float(scatter_val))

        # Find which of these materials exist in the DataFrame
        available_materials = [m for m in material_names if m in absorbance_df.columns]

        if available_materials:
            #  Compute the average absorbance for these materials
            new_data[scatter_key] = absorbance_df[available_materials].mean(axis=1)
            print(f"Scatter volume {scatter_key}: averaged {len(available_materials)} materials")
        else:
            print(f"Warning: No matching materials found for scatter volume {scatter_key}")

    # Create and return the new DataFrame
    averaged_df = pd.DataFrame(new_data)
    return averaged_df


def adjust_scattering(scattering_df, absorbance_df,sample_scat_dict):
    scattering_df = scattering_df.set_index('Wavelength')
    absorbance_df = absorbance_df.set_index('Wavelength')
    result_df = pd.DataFrame(index=absorbance_df.index)

    for adjustment_value, materials in sample_scat_dict.items():
        adjustment_column = str(adjustment_value)  # Convert to string for column name
        for material in materials:
            new_column_name = f"{material}_minus_{adjustment_column}"
            result_df[new_column_name] = absorbance_df[material] - scattering_df[adjustment_column]
    result_df = result_df.reset_index()
    result_df = result_df[order_columns(result_df)]
    return result_df
#def average_scattering_data(dfDataSheet,scatter_map):




grouped_material_pivot_df = create_averaged_material_df(ordered_filtered_pivot_df)
#create_std_df(ordered_filtered_pivot_df)
mat = SpecDataHandler(grouped_material_pivot_df)  # Material_1 will be reference
sam = SpecDataHandler(ordered_filtered_pivot_df)  # First sample will be reference
absorbance_df_sam = create_absorbance_df(sam)
std = create_std_material_df(absorbance_df_sam)
absorbance_df_mat = create_absorbance_df(mat)
scatter_map = create_scattering_map(scatteringInfoSheet_df)
#scatter_df_avg = average_absorbance_by_scatter(scatteringDataSheet_df, scatter_map)
#absorbance_df_scattering_dict = create_scattering_map(sampleInfoSheet_df)
#adjusted_absorbance_df_mat = adjust_scattering(scatter_df_avg, absorbance_df_mat, absorbance_df_scattering_dict)
#adjusted_absorbance_df_sam = adjust_scattering(scatter_df_avg, absorbance_df_sam, absorbance_df_scattering_dict)

'''
print('averaged by scattering volume')
print(scatter_df_avg)

print("dict for input sample scattering vol")
print(absorbance_df_scattering_dict)
print("sample adjusted")
print(adjusted_absorbance_df_sam)

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

print("scattering")
print(scatter_map)

print("scattering info sheet")
print(scatteringInfoSheet_df)

#print("scattering data sheet")
#print(scatteringDataSheet_df)

print("scatter map")
print(scatter_map)

print('averaged by scattering volume')
print(scatter_df_avg)

print("dict for input sample scattering vol")
print(absorbance_df_scattering_dict)

print("adjusted input sample scattering vol")
print(adjusted_absorbance_df_mat)
'''
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
    #adjusted_absorbance_df_mat.to_excel(writer, sheet_name='Adjusted Absorbance Material', index=False)
