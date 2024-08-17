# evaluate_model.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(X_test_csv, y_test_csv, model_file, predictions_file, evaluation_file):
    """
    Evaluate the trained model using test data, and save predictions and evaluation metrics.
    
    :param X_test_csv: Path to the CSV file containing the test features
    :param y_test_csv: Path to the CSV file containing the test target values
    :param model_file: Path to the saved trained model
    :param predictions_file: Path to save the predictions CSV file
    :param evaluation_file: Path to save the evaluation metrics file
    :return: None
    """
    # Load the test data
    X_test = pd.read_csv(X_test_csv, index_col='Datetime')
    y_test = pd.read_csv(y_test_csv, index_col='Datetime')
    
    y_test = y_test.squeeze() # Convert DataFrame to Series if it has only one column

    # Load the trained model
    model = joblib.load(model_file)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save the predictions to a CSV file
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(predictions_file)
    print(f"Predictions have been saved to {predictions_file}")
    
    # Save the evaluation metrics to a file
    evaluation_results = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-squared': r2
    }
    with open(evaluation_file, 'w') as f:
        for key, value in evaluation_results.items():
            f.write(f"{key}: {value}\n")
    print(f"Evaluation metrics have been saved to {evaluation_file}")

if __name__ == "__main__":
    # Define file paths
    X_test_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\X_test.csv'
    y_test_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\y_test.csv'
    model_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\model\linear_regression_model.pkl'
    predictions_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\model_predictions.csv'
    evaluation_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\evaluation_metrics.txt'
    
    # Evaluate the model
    evaluate_model(X_test_file, y_test_file, model_file, predictions_file, evaluation_file)
