import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytz import all_timezones


class SpecDataHandler:
    def __init__(self, pivot_df):
        """
        Initialize with a pivot DataFrame containing Wavelength as index and samples as columns.
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
        Calculate the absorption for a specific sample at a given wavelength using Beer-Lambert's law.

        Parameters:
        sample_name (str): The name of the sample whose absorption is to be calculated.
        wavelength (float): The wavelength for which the absorption is to be calculated.
        zero (str, optional): The name of the reference (default is "Material_1").

        Returns:
        float: The calculated absorbance value.

        Raises:
        ValueError: If the sample or wavelength is not found in the data.
        """
        # Check if sample and zero exist in data
        if sample_name not in self.pivot_df.columns:
            raise ValueError(f"Sample '{sample_name}' not found in data columns.")
        if self.zero not in self.pivot_df.columns:
            raise ValueError(f"Reference '{self.zero}' not found in data columns.")

        # Get sample and reference data for the given wavelength
        sample_data = self.get_sample_data(sample_name)
        zero_data = self.get_sample_data(self.zero)

        # Filter by wavelength
        sample_value = sample_data.loc[sample_data['Wavelength'] == wavelength, sample_name]
        zero_value = zero_data.loc[zero_data['Wavelength'] == wavelength, self.zero]

        # Check if data was found for the given wavelength
        if sample_value.empty:
            raise ValueError(f"Wavelength '{wavelength}' not found for sample '{sample_name}'.")
        if zero_value.empty:
            raise ValueError(f"Wavelength '{wavelength}' not found for reference '{self.zero}'.")

        # Extract the actual values
        sample_value = sample_value.values[0]
        zero_value = zero_value.values[0]

        # Ensure sample value is not zero to avoid division by zero
        if sample_value == 0:
            raise ValueError(
                f"Sample intensity for '{sample_name}' at wavelength '{wavelength}' is zero, leading to a division error.")

        # Calculate absorbance
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
