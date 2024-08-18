# read_data.py

import pandas as pd
import os

def load_and_explore_data(file_path, output_dir, file_name):
    """
    Load the data from a .txt file, explore basic properties, and save it to a CSV file.
    
    :param file_path: Path to the input .txt file
    :param output_dir: Directory to save the output CSV file
    :param file_name: Name of the output CSV file
    :return: None
    """
    # Load the data
    data = pd.read_csv(file_path, sep=';', low_memory=False)
    
    # Display the shape of the data
    print("Shape of the data:", data.shape)
    
    # Display data types of each column
    print("\nData types:\n", data.dtypes)
    
    # Display number of missing values in each column
    print("\nMissing values:\n", data.isnull().sum())
    
    # Create the full output path
    output_path = os.path.join(output_dir, file_name)
    
    # Save the data to a CSV file
    data.to_csv(output_path, index=False)
    print(f"\nData has been saved to {output_path}")

if __name__ == "__main__":
    # Define the file paths and directory
    input_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\datasets\household_power_consumption.txt'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\data'
    output_file_name = 'raw_household_power_consumption.csv'
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load and explore the data
    load_and_explore_data(input_file, output_directory, output_file_name)
