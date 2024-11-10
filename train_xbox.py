import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('pose_controller_data.csv')

# Define the controller axes columns
controller_fieldnames = ['axis_0', 'axis_1', 'axis_2', 'axis_3', 'axis_4', 'axis_5']

# Define the pose landmarks columns
pose_fieldnames = [col for col in data.columns if col not in controller_fieldnames]

# Extract input features (pose landmarks) and output labels (controller axes)
X = data[pose_fieldnames].values
y = data[controller_fieldnames].values

# Include historical data (previous 2 frames)
sequence_length = 3  # Current frame + 2 previous frames

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        # Get the sequence of pose landmarks
        Xs.append(X[i:(i + seq_length)])
        # The corresponding output is the controller axes at the last frame of the sequence
        ys.append(y[i + seq_length - 1])
    return np.array(Xs), np.array(ys)

# Create sequences
X_seq, y_seq = create_sequences(X, y, sequence_length)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42)

# Normalize the input features
from sklearn.preprocessing import StandardScaler

# Reshape X_train and X_val to 2D arrays for scaling
num_samples_train, seq_len, num_features = X_train.shape
num_samples_val = X_val.shape[0]

X_train_reshaped = X_train.reshape(-1, num_features)
X_val_reshaped = X_val.reshape(-1, num_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_val_scaled = scaler.transform(X_val_reshaped)

# Reshape back to sequences
X_train_scaled = X_train_scaled.reshape(num_samples_train, seq_len, num_features)
X_val_scaled = X_val_scaled.reshape(num_samples_val, seq_len, num_features)

# Build the model using LSTM to handle sequential data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.LSTM(128, activation='relu', input_shape=(sequence_length, num_features)),
    layers.Dense(64, activation='relu'),
    layers.Dense(y_train.shape[1])  # Output layer for controller axes
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val_scaled, y_val)
)

# Evaluate the model
loss, mae = model.evaluate(X_val_scaled, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation MAE: {mae}")

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the scaler and the model
import joblib

joblib.dump(scaler, 'scaler.save')
model.save('pose_model.h5')
