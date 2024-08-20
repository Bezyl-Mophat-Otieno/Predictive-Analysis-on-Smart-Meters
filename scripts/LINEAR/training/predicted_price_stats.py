import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_price_predictions(predictions_csv, output_dir, price_per_kwh):
    """
    Generate price predictions based on power consumption and plot the results.
    
    :param predictions_csv: Path to the CSV file containing actual and predicted power consumption with a Datetime index
    :param output_dir: Directory to save the price predictions and plots
    :param price_per_kwh: The price per kWh (in KSH) to use for calculating costs
    :return: None
    """
    # Check if the CSV file exists
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"CSV file not found: {predictions_csv}")
    
    # Load the predictions
    data = pd.read_csv(predictions_csv, index_col='Datetime', parse_dates=True)
    
    # Check if the expected columns are present
    if 'Actual' not in data.columns or 'Predicted' not in data.columns:
        raise ValueError("CSV file must contain 'Actual' and 'Predicted' columns.")
    
    # Convert from Wh to kWh for better readability in plots
    data['Actual_kWh'] = data['Actual'] / 1000.0
    data['Predicted_kWh'] = data['Predicted'] / 1000.0
    
    # Calculate actual and predicted prices using the constant price per kWh
    data['Actual_Cost'] = data['Actual_kWh'] * price_per_kwh
    data['Predicted_Cost'] = data['Predicted_kWh'] * price_per_kwh
    
    # Extract relevant time features
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    
    # Aggregate price predictions by hour of the day
    hourly_stats = data.groupby('Hour').agg({
        'Actual_Cost': ['mean', 'sum'],
        'Predicted_Cost': ['mean', 'sum']
    })
    hourly_stats.columns = ['_'.join(col) for col in hourly_stats.columns]
    
    # Aggregate price predictions by day of the week
    daily_stats = data.groupby('DayOfWeek').agg({
        'Actual_Cost': ['mean', 'sum'],
        'Predicted_Cost': ['mean', 'sum']
    })
    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
    
    # Aggregate price predictions by month of the year
    monthly_stats = data.groupby('Month').agg({
        'Actual_Cost': ['mean', 'sum'],
        'Predicted_Cost': ['mean', 'sum']
    })
    monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save price statistics to CSV files
    hourly_stats.to_csv(os.path.join(output_dir, 'hourly_price_statistics.csv'))
    daily_stats.to_csv(os.path.join(output_dir, 'daily_price_statistics.csv'))
    monthly_stats.to_csv(os.path.join(output_dir, 'monthly_price_statistics.csv'))
    
    # Print the price stats to the console
    print("Hourly Price Statistics:\n", hourly_stats)
    print("Daily Price Statistics:\n", daily_stats)
    print("Monthly Price Statistics:\n", monthly_stats)
    
    # Plotting the price statistics
    plt.figure(figsize=(14, 12))  # Increase the figure size

    # Plot hourly price statistics
    plt.subplot(3, 1, 1)
    plt.plot(hourly_stats.index, hourly_stats['Actual_Cost_mean'], label='Actual Cost', color='blue')
    plt.plot(hourly_stats.index, hourly_stats['Predicted_Cost_mean'], label='Predicted Cost', color='red', linestyle='--')
    plt.title('Average Cost by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Cost (KSH)')
    plt.xticks(fontsize=10)  # Adjust x-axis label size
    plt.yticks(fontsize=10)  # Adjust y-axis label size
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Plot daily price statistics
    plt.subplot(3, 1, 2)
    plt.plot(daily_stats.index, daily_stats['Actual_Cost_mean'], label='Actual Cost', color='blue')
    plt.plot(daily_stats.index, daily_stats['Predicted_Cost_mean'], label='Predicted Cost', color='red', linestyle='--')
    plt.title('Average Cost by Day of the Week')
    plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
    plt.ylabel('Cost (KSH)')
    plt.xticks(fontsize=10)  # Adjust x-axis label size
    plt.yticks(fontsize=10)  # Adjust y-axis label size
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Plot monthly price statistics
    plt.subplot(3, 1, 3)
    plt.plot(monthly_stats.index, monthly_stats['Actual_Cost_mean'], label='Actual Cost', color='blue')
    plt.plot(monthly_stats.index, monthly_stats['Predicted_Cost_mean'], label='Predicted Cost', color='red', linestyle='--')
    plt.title('Average Cost by Month of the Year')
    plt.xlabel('Month')
    plt.ylabel('Cost (KSH)')
    plt.xticks(fontsize=10, rotation=45)  # Adjust x-axis label size and rotation
    plt.yticks(fontsize=10)  # Adjust y-axis label size
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout(pad=4.0)  # Add padding between subplots
    plt.savefig(os.path.join(output_dir, 'price_statistics.png'))
    plt.show()
    print("Plots saved as 'price_statistics.png'")

if __name__ == "__main__":
    # Define file paths
    predictions_file = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\data\model_predictions.csv'
    output_directory = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\plots\price_statistics'
    
    # Set price per kWh
    price_per_kwh = 23.25  # KSH per kWh
    
    # Generate price predictions and plots
    generate_price_predictions(predictions_file, output_directory, price_per_kwh)
