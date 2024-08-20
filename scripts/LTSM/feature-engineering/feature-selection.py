# scripts/feature_engineering.py

import pandas as pd
import numpy as np

# Load the cleaned dataset
data_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_cleaned.csv"
data_cleaned = pd.read_csv(data_path, sep=',', low_memory=False)

# Parse Date and Time columns
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%Y-%m-%d')
data_cleaned['Time'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M:%S').dt.time

# Feature Engineering
data_cleaned['Hour'] = data_cleaned['Time'].apply(lambda x: x.hour)
data_cleaned['Minute'] = data_cleaned['Time'].apply(lambda x: x.minute)
data_cleaned['Month'] = data_cleaned['Date'].dt.month
data_cleaned['DayOfWeek'] = data_cleaned['Date'].dt.dayofweek
data_cleaned['IsWeekend'] = data_cleaned['DayOfWeek'] >= 5

# Aggregate Features
data_cleaned['Daily_Avg'] = data_cleaned.groupby('Date')['Global_active_power'].transform('mean')
data_cleaned['Weekly_Avg'] = data_cleaned['Daily_Avg'].rolling(window=7).mean()
data_cleaned['Monthly_Avg'] = data_cleaned['Daily_Avg'].rolling(window=30).mean()

# Statistical Features
data_cleaned['Global_active_power_Mean'] = data_cleaned['Global_active_power'].mean()
data_cleaned['Global_active_power_Std'] = data_cleaned['Global_active_power'].std()

# Lag Features
data_cleaned['Lag_1'] = data_cleaned['Global_active_power'].shift(1)
data_cleaned['Lag_24'] = data_cleaned['Global_active_power'].shift(24)

# Drop NaN values caused by lagging
data_cleaned.dropna(inplace=True)

# Save the dataset with engineered features
output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_features.csv'
data_cleaned.to_csv(output_path, index=False)
