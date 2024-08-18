# scripts/sequence_preparation.py

import pandas as pd
import numpy as np
import os

def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])  # All features except the target
        y.append(data[i + sequence_length, -1])  # Only the target
    return np.array(X), np.array(y)

def prepare_sequences(features_path, target_path, sequence_length, output_path):
    # Load the cleaned and scaled dataset
    scaled_features = pd.read_csv(features_path).values
    scaled_target = pd.read_csv(target_path).values

    # Combine features and target into one dataset
    data = np.hstack((scaled_features, scaled_target))

    # Create sequences
    X, y = create_sequences(data, sequence_length)

    # Reshape data for LSTM
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # Save prepared sequences
    np.savez(output_path, X=X, y=y)
    print(f"Sequences saved to {output_path}")

if __name__ == "__main__":
    # Define parameters for sequence preparation
    sequence_length = 24  # Example sequence length, can be adjusted based on your use case

    # Prepare training sequences
    train_features_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_scaled_features.csv'
    train_target_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\train_scaled_target.csv'
    train_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\sequences\train_sequences.npz'
    prepare_sequences(train_features_path, train_target_path, sequence_length, train_output_path)

    # Prepare test sequences
    test_features_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_scaled_features.csv'
    test_target_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_scaled_target.csv'
    test_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\sequences\test_sequences.npz'
    prepare_sequences(test_features_path, test_target_path, sequence_length, test_output_path)
