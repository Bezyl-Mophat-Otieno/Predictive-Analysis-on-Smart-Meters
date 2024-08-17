# scripts/model_building.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the sequences
data_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\outputs\sequences\train_sequences.npz"
data = np.load(data_path)
X, y = data['X'], data['y']

# Split the data into training and validation sets
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping]
)

# Save the model
model.save(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\models\lstm_new_model.h5')

print("Model training complete and saved.")
