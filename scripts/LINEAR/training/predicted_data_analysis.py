import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_statistics(predictions_csv, output_dir):
    """
    Generate statistics and insights based on the model's predictions.
    
    :param predictions_csv: Path to the CSV file containing actual and predicted values with a Datetime index
    :param output_dir: Directory to save the statistical outputs and plots
    :return: None
    """
    # Load the predictions
    data = pd.read_csv(predictions_csv, index_col='Datetime', parse_dates=True)
    
    # Extract relevant time features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    
    # Convert from Wh to kWh for better readability in plots
    data['Actual_kWh'] = data['Actual'] / 1000.0
    data['Predicted_kWh'] = data['Predicted'] / 1000.0
    
    # Aggregate predictions by hour of the day
    hourly_stats = data.groupby('Hour').agg({
        'Actual_kWh': ['mean', 'sum'],
        'Predicted_kWh': ['mean', 'sum']
    })
    hourly_stats.columns = ['_'.join(col) for col in hourly_stats.columns]
    
    # Aggregate predictions by day of the week
    daily_stats = data.groupby('DayOfWeek').agg({
        'Actual_kWh': ['mean', 'sum'],
        'Predicted_kWh': ['mean', 'sum']
    })
    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
    
    # Aggregate predictions by month of the year
    monthly_stats = data.groupby('Month').agg({
        'Actual_kWh': ['mean', 'sum'],
        'Predicted_kWh': ['mean', 'sum']
    })
    monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save statistics to CSV files
    hourly_stats.to_csv(os.path.join(output_dir, 'hourly_statistics.csv'))
    daily_stats.to_csv(os.path.join(output_dir, 'daily_statistics.csv'))
    monthly_stats.to_csv(os.path.join(output_dir, 'monthly_statistics.csv'))
    
    # Print the stats to the console
    print("Hourly Statistics:\n", hourly_stats)
    print("Daily Statistics:\n", daily_stats)
    print("Monthly Statistics:\n", monthly_stats)
    
    # Plotting the statistics
    plt.figure(figsize=(12, 8))

    # Plot hourly statistics
    plt.subplot(3, 1, 1)
    plt.plot(hourly_stats.index, hourly_stats['Actual_kWh_mean'], label='Actual', color='blue')
    plt.plot(hourly_stats.index, hourly_stats['Predicted_kWh_mean'], label='Predicted', color='red', linestyle='--')
    plt.title('Average Energy Consumption (kWh) by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    # Plot daily statistics
    plt.subplot(3, 1, 2)
    plt.plot(daily_stats.index, daily_stats['Actual_kWh_mean'], label='Actual', color='blue')
    plt.plot(daily_stats.index, daily_stats['Predicted_kWh_mean'], label='Predicted', color='red', linestyle='--')
    plt.title('Average Energy Consumption (kWh) by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    # Plot monthly statistics
    plt.subplot(3, 1, 3)
    plt.plot(monthly_stats.index, monthly_stats['Actual_kWh_mean'], label='Actual', color='blue')
    plt.plot(monthly_stats.index, monthly_stats['Predicted_kWh_mean'], label='Predicted', color='red', linestyle='--')
    plt.title('Average Energy Consumption (kWh) by Month of the Year')
    plt.xlabel('Month')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(pad=4.0)  # Add padding between subplots
    plt.savefig(os.path.join(output_dir, 'energy_consumption_statistics.png'))
    plt.show()
    print("Plots saved as 'energy_consumption_statistics.png'")

if __name__ == "__main__":
    # Define file paths
    predictions_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\data\model_predictions.csv'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\plots'
    
    # Generate statistics and plots
    generate_statistics(predictions_file, output_directory)
