# scripts/data_scaling.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Load the training dataset
train_features_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_features.csv"
train_target_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_target.csv"

train_features = pd.read_csv(train_features_path, sep=',', low_memory=False)
train_target = pd.read_csv(train_target_path, sep=',', low_memory=False)

# Define features and target
features = ['Voltage', 'Global_intensity', 'Sub_metering_1', 
            'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'Month', 'Daily_Avg', 'Weekly_Avg', 'Monthly_Avg']
target = 'Global_active_power'

# Prepare features and target arrays
X_train = train_features[features].values
y_train = train_target[target].values

# Choose a scaler
scaler_X = MinMaxScaler()  # You could also use StandardScaler()
scaler_y = MinMaxScaler()

# Fit and transform the features
X_train_scaled = scaler_X.fit_transform(X_train)

# Transform the target
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

# Create DataFrames for scaled features and target
data_scaled_features = pd.DataFrame(X_train_scaled, columns=features)
data_scaled_target = pd.DataFrame(y_train_scaled, columns=[target])

# Save the scaled features and target separately
features_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_scaled_features.csv'
target_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_scaled_target.csv'

data_scaled_features.to_csv(features_output_path, index=False)
data_scaled_target.to_csv(target_output_path, index=False)

# Save the scalers
scaler_X_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scalers\scaler_X.pkl'
scaler_y_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scalers\scaler_y.pkl'

joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)

print(f"Scaled training features saved to {features_output_path}")
print(f"Scaled training target saved to {target_output_path}")
print(f"Scalers saved to {scaler_X_path} and {scaler_y_path}")
