# scripts/test_data_scaling.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Load the test dataset
test_features_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_features.csv"
test_target_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_target.csv"

test_features = pd.read_csv(test_features_path, sep=',', low_memory=False)
test_target = pd.read_csv(test_target_path, sep=',', low_memory=False)

# Define features and target
features = ['Voltage', 'Global_intensity', 'Sub_metering_1', 
            'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'Month', 'Daily_Avg', 'Weekly_Avg', 'Monthly_Avg']
target = 'Global_active_power'

# Prepare features and target arrays
X_test = test_features[features].values
y_test = test_target[target].values

# Load the pre-fitted scalers
scaler_X_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scalers\scaler_X.pkl'
scaler_y_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scalers\scaler_y.pkl'

scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# Transform the features using the pre-fitted scaler
X_test_scaled = scaler_X.transform(X_test)

# Transform the target using the pre-fitted scaler
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Create DataFrames for scaled features and target
data_scaled_features = pd.DataFrame(X_test_scaled, columns=features)
data_scaled_target = pd.DataFrame(y_test_scaled, columns=[target])

# Save the scaled features and target separately
features_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_scaled_features.csv'
target_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_scaled_target.csv'

data_scaled_features.to_csv(features_output_path, index=False)
data_scaled_target.to_csv(target_output_path, index=False)

print(f"Scaled test features saved to {features_output_path}")
print(f"Scaled test target saved to {target_output_path}")
