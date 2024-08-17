# preprocess_data.py

import pandas as pd
import os

def preprocess_data(input_csv, output_dir, file_name):
    """
    Preprocess the raw data by combining Date and Time into a single Datetime column,
    handling missing values, and saving the preprocessed data.
    
    :param input_csv: Path to the input raw CSV file
    :param output_dir: Directory to save the output CSV file
    :param file_name: Name of the output CSV file
    :return: None
    """
    # Load the raw data
    data = pd.read_csv(input_csv, low_memory=False)
    
    # Combine Date and Time into a single datetime column
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Set the datetime as the index
    data.set_index('Datetime', inplace=True)
    
    # Drop the original Global_reactive_power, Sub_metering_3, Date, and Time columns
    data.drop(['Date', 'Time', 'Global_reactive_power', 'Sub_metering_3'], axis=1, inplace=True)
    
    # Rename columns to simpler names
    data.rename(columns={
        'Global_active_power': 'Power',
        'Voltage': 'Voltage',
        'Global_intensity': 'Current',
        'Sub_metering_1': 'Load_1',
        'Sub_metering_2': 'Load_2',
    }, inplace=True)
    
    # Handle non-numeric values by converting all columns to numeric and coercing errors
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Handle missing values by removing rows with any missing values
    data_cleaned = data.dropna()
    
    # Display the shape of the cleaned data
    print("Shape of the cleaned data:", data_cleaned.shape)
    
    # Create the full output path
    output_path = os.path.join(output_dir, file_name)

    # Save the preprocessed data to a CSV file
    data_cleaned.to_csv(output_path, index=True)
    print(f"\nPreprocessed data has been saved to {output_path}")

if __name__ == "__main__":
    # Define the input file, output directory, and output file name
    input_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\raw_household_power_consumption.csv'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data'
    output_file_name = 'preprocessed_household_power_consumption.csv'
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Preprocess the data
    preprocess_data(input_file, output_directory, output_file_name)
