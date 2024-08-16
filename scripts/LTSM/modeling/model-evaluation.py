# scripts/model_evaluation.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the scaled test data
X_test_scaled = pd.read_csv(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_scaled_features.csv').values
y_test = pd.read_csv(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\test_scaled_target.csv').values

# Load the saved scalers
scaler_X = joblib.load(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scalers\test-scaler_X.pkl')
scaler_y = joblib.load(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scalers\test-scaler_y.pkl')

# Define sequence length (must match the sequence length used during training)
sequence_length = 24

# Calculate the number of samples
num_samples = X_test_scaled.shape[0] // sequence_length

# Reshape to match LSTM input requirements
X_test_reshaped = X_test_scaled[:num_samples * sequence_length].reshape((num_samples, sequence_length, X_test_scaled.shape[1]))

# Flatten y_test to align with the LSTM output
y_test_reshaped = y_test[:num_samples * sequence_length].reshape(num_samples, sequence_length, 1)

print(f"Reshaped X_test shape: {X_test_reshaped.shape}")
print(f"Reshaped y_test shape: {y_test_reshaped.shape}")

# Load the trained model
model_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\models\lstm_new_model.h5'
model = load_model(model_path)

# Make predictions
y_pred_scaled = model.predict(X_test_reshaped)

# Flatten predictions and actuals for comparison
y_pred_scaled_flattened = y_pred_scaled.flatten()
y_test_flattened = y_test_reshaped.flatten()

# Inverse transform the predictions and y_test to get them back to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled_flattened.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test_flattened.reshape(-1, 1))

# Ensure y_test and y_pred are of the same length
min_length = min(len(y_test), len(y_pred))
y_test = y_test[:min_length]
y_pred = y_pred[:min_length]

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the predictions to a CSV file
predictions_output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\data\predictions.csv'
pd.DataFrame(y_pred, columns=['Predicted_Global_active_power']).to_csv(predictions_output_path, index=False)

print(f"Predictions saved to {predictions_output_path}")

# Define the output directory for plots
plots_output_dir = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\plots'
os.makedirs(plots_output_dir, exist_ok=True)

# Plotting

# 1. Line Plot - Actual vs. Predicted
plt.figure(figsize=(15, 6))
plt.plot(y_test[:1000], label='Actual', color='blue')
plt.plot(y_pred[:1000], label='Predicted', color='orange')
plt.title('Actual vs. Predicted Global Active Power')
plt.xlabel('Time Steps')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.savefig(os.path.join(plots_output_dir, 'actual_vs_predicted.png'))
plt.show()
plt.close()

# 2. Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Global Active Power (kilowatts)')
plt.ylabel('Residuals')
plt.savefig(os.path.join(plots_output_dir, 'residual_plot.png'))
plt.show()
plt.close()

# 3. Histogram of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, color='green')
plt.title('Distribution of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plots_output_dir, 'residuals_histogram.png'))
plt.show()
plt.close()
