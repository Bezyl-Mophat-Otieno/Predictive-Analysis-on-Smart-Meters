# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(input_csv, output_dir):
    """
    Perform Exploratory Data Analysis (EDA) on the preprocessed data.
    
    :param input_csv: Path to the input preprocessed CSV file
    :param output_dir: Directory to save the EDA plots
    :return: None
    """
    # Load the preprocessed data
    data = pd.read_csv(input_csv, index_col='Datetime', parse_dates=['Datetime'])
    
    # Convert non-numeric values to NaN
    data.replace('?', pd.NA, inplace=True)
    
    # Convert all columns to numeric types, errors='coerce' will turn non-numeric values into NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Display summary statistics
    print("Summary Statistics:")
    print(data.describe())
    
    # Plotting Global Active Power over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Power'], label='Global Active Power', color='blue')
    plt.xlabel('Datetime')
    plt.ylabel('Power (kW)')
    plt.title('Global Active Power over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'power_over_time.png'))
    plt.close()
    
    # Plotting Current over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Current'], label='Current', color='green')
    plt.xlabel('Datetime')
    plt.ylabel('Current (A)')
    plt.title('Current over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'current_over_time.png'))
    plt.close()
    
    # Plotting Load - Kitchen and Laundry over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Load_1'], label='Load - 1', color='red')
    plt.plot(data.index, data['Load_2'], label='Load - 2', color='orange')
    plt.xlabel('Datetime')
    plt.ylabel('Load (Wh)')
    plt.title('Load - 1 and 2 over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loads_over_time.png'))
    plt.close()

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    print("Exploratory Data Analysis (EDA) has been completed.")
    

if __name__ == "__main__":
    # Define the input file and output directory for EDA plots
    input_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\preprocessed_household_power_consumption.csv'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\plots'
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Perform EDA
    perform_eda(input_file, output_directory)
