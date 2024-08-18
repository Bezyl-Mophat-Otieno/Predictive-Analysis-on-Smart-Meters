# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_and_save_model(X_train_csv, y_train_csv, model_output):
    """
    Train a linear regression model and save it to a file.
    
    :param X_train_csv: Path to the training features CSV file
    :param y_train_csv: Path to the training target CSV file
    :param model_output: Path to save the trained model
    :return: None
    """
    # Load training data
    X_train = pd.read_csv(X_train_csv, index_col='Datetime', parse_dates=['Datetime'])
    
    # Load the target data
    y_train = pd.read_csv(y_train_csv, index_col='Datetime')
    
    # Extract the target variable as a Series
    y_train = y_train.squeeze()  # Convert DataFrame to Series if it has only one column
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, model_output)
    
    print(f"Model has been saved to {model_output}")

if __name__ == "__main__":
    # Define file paths
    X_train_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\data\X_train.csv'
    y_train_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\data\y_train.csv'
    model_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\model\linear_regression_model.pkl'
    
    # Train the model and save it
    train_and_save_model(X_train_file, y_train_file, model_file)
