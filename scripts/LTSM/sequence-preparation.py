# scripts/sequence_preparation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load the cleaned and scaled dataset
features_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_scaled_features.csv'
target_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_scaled_target.csv'

scaled_features = pd.read_csv(features_path).values
scaled_target = pd.read_csv(target_path).values

# Combine features and target into one dataset
data = np.hstack((scaled_features, scaled_target))

# Define sequence length
sequence_length = 24  # For example, using the past 24 hours to predict the next value

# Function to create sequences
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])  # All features except the target
        y.append(data[i+sequence_length, -1])  # Only the target
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(data, sequence_length)

# Reshape data for LSTM
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Save prepared sequences
sequences_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_sequences.npz'
np.savez(sequences_output_path, X=X, y=y)

print(f"Sequences saved to {sequences_output_path}")
