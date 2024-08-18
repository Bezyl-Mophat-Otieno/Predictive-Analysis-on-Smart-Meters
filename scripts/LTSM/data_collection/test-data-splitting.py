# scripts/data_splitting.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the cleaned dataset
data_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_features.csv"
data_cleaned = pd.read_csv(data_path, sep=',', low_memory=False)

# Define features and target
features = ['Voltage', 'Global_intensity', 'Sub_metering_1', 
            'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'Month', 'Daily_Avg', 'Weekly_Avg', 'Monthly_Avg']
target = 'Global_active_power'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_cleaned[features], data_cleaned[target], test_size=0.2, random_state=42)

# Save the training and testing data separately
train_features_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_features.csv'
train_target_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_target.csv'

test_features_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_features.csv'
test_target_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_target.csv'

# Create DataFrames for the splits
train_features = pd.DataFrame(X_train, columns=features)
train_target = pd.DataFrame(y_train, columns=[target])

test_features = pd.DataFrame(X_test, columns=features)
test_target = pd.DataFrame(y_test, columns=[target])

# Save to CSV
train_features.to_csv(train_features_output_path, index=False)
train_target.to_csv(train_target_output_path, index=False)

test_features.to_csv(test_features_output_path, index=False)
test_target.to_csv(test_target_output_path, index=False)

print(f"Training features saved to {train_features_output_path}")
print(f"Training target saved to {train_target_output_path}")
print(f"Testing features saved to {test_features_output_path}")
print(f"Testing target saved to {test_target_output_path}")
