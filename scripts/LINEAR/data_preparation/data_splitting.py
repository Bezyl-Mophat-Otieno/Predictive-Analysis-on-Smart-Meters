# data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data(input_csv, output_dir):
    """
    Prepare data for modeling by splitting it into training and testing sets.
    
    :param input_csv: Path to the input preprocessed CSV file
    :param output_dir: Directory to save the training and testing CSV files
    :return: None
    """
    # Load the preprocessed data
    data = pd.read_csv(input_csv, index_col='Datetime', parse_dates=['Datetime'])
    
    # Define the features and target variable
    features = ['Current', 'Voltage', 'Load_Kitchen', 'Load_Laundry']
    target = 'Power'
    
    X = data[features]
    y = data[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the training and testing sets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=True)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=True)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=True)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=True)
    
    print(f"Training and testing data have been saved to {output_dir}")

if __name__ == "__main__":
    # Define the input file and output directory for data preparation
    input_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\data_preparation\data\manipulated_smart_meter_data.csv'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\data_preparation\data'
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Prepare the data
    prepare_data(input_file, output_directory)
