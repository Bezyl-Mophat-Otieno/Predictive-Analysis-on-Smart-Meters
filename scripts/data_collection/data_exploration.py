# scripts/data_exploration.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the path chunksize to handle large plots
plt.rcParams['agg.path.chunksize'] = 10000

# Load the cleaned dataset
data_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_cleaned.csv"
data_cleaned = pd.read_csv(data_path, sep=',', low_memory=False)

# Parse Date and Time columns separately
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%Y-%m-%d')
data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M:%S').dt.time

# Feature engineering
data_cleaned['Hour'] = data_cleaned['Time'].apply(lambda x: x.hour)
data_cleaned['Minute'] = data_cleaned['Time'].apply(lambda x: x.minute)
data_cleaned['DayOfWeek'] = data_cleaned['Date'].dt.dayofweek
data_cleaned['IsWeekend'] = data_cleaned['DayOfWeek'] >= 5

# Sample 10% of the data for quicker processing
data_sampled = data_cleaned.sample(frac=0.1, random_state=1)

# Create output directory if it doesn't exist
output_dir = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\images"
os.makedirs(output_dir, exist_ok=True)

# Plotting Global Active Power over time
plt.figure(figsize=(15, 5))
plt.plot(data_sampled['Date'], data_sampled['Global_active_power'])
plt.title('Global Active Power over Time')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated labels
plt.savefig(os.path.join(output_dir, 'global_active_power_over_time.png'))
plt.close()  # Close the plot to free memory

# Summary statistics
print(data_sampled.describe())

# Histograms of numerical columns
data_sampled.hist(figsize=(15, 10), bins=50)
plt.savefig(os.path.join(output_dir, 'histograms.png'))
plt.close()  # Close the plot to free memory

# Drop non-numeric columns for correlation matrix
data_corr = data_sampled.drop(['Date', 'Time'], axis=1)

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()  # Close the plot to free memory
