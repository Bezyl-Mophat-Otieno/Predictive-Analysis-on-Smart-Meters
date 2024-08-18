import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_power_statistics(predictions_csv, output_dir):
    """
    Generate and plot power consumption statistics based on predictions.
    
    :param predictions_csv: Path to the CSV file containing actual and predicted values with a Datetime index
    :param output_dir: Directory to save the power consumption statistics and plots
    :return: None
    """
    # Load the predictions
    data = pd.read_csv(predictions_csv, index_col='Datetime', parse_dates=True)
    
    # Extract relevant time features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    
    # Aggregate power predictions by hour of the day
    hourly_stats = data.groupby('Hour').agg({
        'Actual': ['mean', 'sum'],
        'Predicted': ['mean', 'sum']
    })
    hourly_stats.columns = ['_'.join(col) for col in hourly_stats.columns]
    
    # Aggregate power predictions by day of the week
    daily_stats = data.groupby('DayOfWeek').agg({
        'Actual': ['mean', 'sum'],
        'Predicted': ['mean', 'sum']
    })
    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
    
    # Aggregate power predictions by month of the year
    monthly_stats = data.groupby('Month').agg({
        'Actual': ['mean', 'sum'],
        'Predicted': ['mean', 'sum']
    })
    monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save power statistics to CSV files
    hourly_stats.to_csv(os.path.join(output_dir, 'hourly_power_statistics.csv'))
    daily_stats.to_csv(os.path.join(output_dir, 'daily_power_statistics.csv'))
    monthly_stats.to_csv(os.path.join(output_dir, 'monthly_power_statistics.csv'))
    
    # Print the power stats to the console
    print("Hourly Power Statistics:\n", hourly_stats)
    print("Daily Power Statistics:\n", daily_stats)
    print("Monthly Power Statistics:\n", monthly_stats)
    
    # Plotting the power statistics
    plt.figure(figsize=(12, 8))

    # Plot hourly power statistics
    plt.subplot(3, 1, 1)
    plt.plot(hourly_stats.index, hourly_stats['Actual_mean'], label='Actual Power', color='blue')
    plt.plot(hourly_stats.index, hourly_stats['Predicted_mean'], label='Predicted Power', color='red', linestyle='--')
    plt.title('Average Power Consumption by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Power Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    # Plot daily power statistics
    plt.subplot(3, 1, 2)
    plt.plot(daily_stats.index, daily_stats['Actual_mean'], label='Actual Power', color='blue')
    plt.plot(daily_stats.index, daily_stats['Predicted_mean'], label='Predicted Power', color='red', linestyle='--')
    plt.title('Average Power Consumption by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
    plt.ylabel('Power Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    # Plot monthly power statistics
    plt.subplot(3, 1, 3)
    plt.plot(monthly_stats.index, monthly_stats['Actual_mean'], label='Actual Power', color='blue')
    plt.plot(monthly_stats.index, monthly_stats['Predicted_mean'], label='Predicted Power', color='red', linestyle='--')
    plt.title('Average Power Consumption by Month of the Year')
    plt.xlabel('Month')
    plt.ylabel('Power Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(pad=4.0)  # Add padding between subplots
    plt.savefig(os.path.join(output_dir, 'power_statistics.png'))
    plt.show()
    print("Plots saved as 'power_statistics.png'")

if __name__ == "__main__":
    # Define file paths
    predictions_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\outputs\data\model_predictions.csv'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\outputs\power_statistics'
    
    # Generate power statistics and plots
    generate_power_statistics(predictions_file, output_directory)
