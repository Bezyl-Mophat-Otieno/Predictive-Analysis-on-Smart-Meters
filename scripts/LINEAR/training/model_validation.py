# validate_model_sampled.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

def plot_model_validation_sampled(predictions_csv, sample_size=1000):
    """
    Plot various metrics to validate the model's predictions against the actual values using a sample of the data.
    
    :param predictions_csv: Path to the CSV file containing the actual and predicted values
    :param sample_size: Number of samples to plot
    :return: None
    """
    # Load the predictions data
    data = pd.read_csv(predictions_csv, index_col='Datetime')
    
    # Sample the data if it's too large
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42).sort_index()
    
    # Calculate residuals
    data['Residuals'] = data['Actual'] - data['Predicted']

    # Plot residuals over time
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Residuals'], label='Residuals', color='purple')
    plt.xlabel('Datetime')
    plt.ylabel('Residuals')
    plt.title('Residuals Over Time (Sampled)')
    plt.axhline(y=0, color='red', linestyle='--', lw=1)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('residuals_over_time_sampled.png')
    plt.show()

    # Plot histogram of residuals
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Residuals'], kde=True, color='blue', bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals (Sampled)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('residuals_distribution_sampled.png')
    plt.show()

    # QQ plot of residuals
    plt.figure(figsize=(8, 6))
    stats.probplot(data['Residuals'], dist="norm", plot=plt)
    plt.title('QQ Plot of Residuals (Sampled)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('qq_plot_residuals_sampled.png')
    plt.show()

    # Scatter plot of Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Actual'], data['Predicted'], color='green', edgecolor='k', alpha=0.7)
    plt.plot([data['Actual'].min(), data['Actual'].max()], 
             [data['Actual'].min(), data['Actual'].max()], 
             color='red', lw=2, linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values (Sampled)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_sampled.png')
    plt.show()

if __name__ == "__main__":
    # Define the path to the predictions CSV file
    predictions_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\outputs\data\model_predictions.csv'
    
    # Plot the model validation metrics using a sample
    plot_model_validation_sampled(predictions_file, sample_size=1000)
