"""
SpecDataHandler Class
===================

A class for handling and processing spectroscopy data with built-in absorption calculations
and visualization capabilities.

Key Features:
    - Manages spectroscopy data with wavelength and intensity measurements
    - Calculates absorption using reference measurements
    - Provides data visualization tools
    - Supports both single-sample and batch processing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytz import all_timezones


class SpecDataHandler:
    def __init__(self, pivot_df):
        """
        Initialize the SpecDataHandler with spectroscopy data.

        Parameters:
            pivot_df (pd.DataFrame): DataFrame containing:
                - 'Wavelength' column
                - Sample columns with intensity measurements
                First column (Material_1) is assumed to be the reference measurement.
        """
        self.pivot_df = pivot_df
        self.samples = [col for col in pivot_df.columns if col != 'Wavelength']
        self.zero = pivot_df.columns[1]

    def get_sample_data(self, sample_name):
        """
        Get the Wavelength and Intensity data for a specific sample.
        """
        if sample_name in self.samples:
            return self.pivot_df[['Wavelength', sample_name]]
        else:
            raise ValueError(f"Sample '{sample_name}' not found in the data.")

    def print_stats(self):
        print(f"Current Zero \n {self.zero}")
        print(f"Current Data \n {self.pivot_df}")
    def return_all(self):
        return self.pivot_df
    def set_zero(self, zero):
        self.zero = zero

    def get_one_absorption(self, sample_name, wavelength):
        """
        Calculate absorbance for a single sample at a specific wavelength.

        Parameters:
            sample_name (str): Name of the sample column
            wavelength (float): Target wavelength for calculation

        Returns:
            float: Calculated absorbance value using Beer-Lambert law:
                  A = log(I₀/I), where I₀ is reference intensity and I is sample intensity

        Raises:
            ValueError: If sample not found or intensity is zero
        """
        if sample_name not in self.pivot_df.columns:
            raise ValueError(f"Sample '{sample_name}' not found in data columns.")

        # Get sample data
        sample_data = self.get_sample_data(sample_name)
        
        # Get reference value at this wavelength
        zero_value = self.pivot_df.loc[self.pivot_df['Wavelength'] == wavelength, self.zero].values[0]
        sample_value = sample_data.loc[sample_data['Wavelength'] == wavelength, sample_name].values[0]

        # Calculate absorbance using averaged reference
        if sample_value == 0:
            raise ValueError(f"Sample intensity is zero, leading to a division error.")
        
        absorbance = np.log(zero_value / sample_value)
        return absorbance

    def all_absorbance(self, sample_name):
        """
        Calculate the absorption for a specific sample across all available wavelengths.

        Parameters:
        sample_name (str): The name of the sample for which absorption is to be calculated.
        zero (str, optional): The name of the reference (default is "Material_1").

        Returns:
        pd.DataFrame: A DataFrame with columns 'Wavelength' and 'Absorbance'.
        """
        # Get unique wavelengths to avoid redundant calculations
        wavelengths = self.pivot_df['Wavelength'].unique()

        # List to store results
        absorbance_results = []

        # Iterate over each unique wavelength and calculate absorbance
        for wavelength in wavelengths:
            try:
                result = self.get_one_absorption(sample_name, wavelength)
                absorbance_results.append({'Wavelength': wavelength, 'Absorbance': result})
            except ValueError as e:
                print(f"Warning: {e}")
        absorbance_df = pd.DataFrame(absorbance_results)

        return absorbance_df

    def dataset_absorbance(self):
        absorbance_results = []
        # Iterate over each sample in the DataFrame columns (excluding 'Wavelength')
        for sample in self.pivot_df.columns:
            if sample != 'Wavelength':
                current_absorbance = self.all_absorbance(sample)
                if isinstance(current_absorbance, pd.DataFrame):
                    for _, row in current_absorbance.iterrows():
                        absorbance_results.append(
                            {'Sample': sample, 'Wavelength': row['Wavelength'], 'Absorbance': row['Absorbance']})

        # Convert results to DataFrame
        dataset_absorbance_df = pd.DataFrame(absorbance_results)
        pivot_df = dataset_absorbance_df.pivot(index='Wavelength', columns='Sample', values='Absorbance')

        return pivot_df

    def plot_sample(self, sample_name):
        """
        Plot the absorption data for a specific sample.
        """
        if sample_name in self.samples:
            plt.figure(figsize=(10, 6))
            plt.plot(self.pivot_df['Wavelength'], self.pivot_df[sample_name], label=sample_name)
            plt.xlabel('Wavelength')
            plt.ylabel('Absorption')
            plt.title(f'Absorption Spectrum for {sample_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            raise ValueError(f"Sample '{sample_name}' not found in the data.")
