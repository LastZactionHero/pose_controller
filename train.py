
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('pose_training_data.csv')

# Drop unnecessary columns
data = data.drop(columns=['frame', 'label'])

# Separate input features and output labels
X = data.drop(columns=['roll', 'pitch']).values  # Input features
y = data[['roll', 'pitch']].values  # Output labels

# Normalize the input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optionally, normalize the output labels (if needed)
# For this case, we'll assume roll and pitch are in degrees (0-180), so scaling might help
from sklearn.preprocessing import MinMaxScaler
label_scaler = MinMaxScaler()
y_scaled = label_scaler.fit_transform(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)  # Output layer for roll and pitch
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Evaluate the model
loss, mae = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation MAE: {mae}")

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

import joblib

# Save the scalers
joblib.dump(scaler, 'scaler.save')
joblib.dump(label_scaler, 'label_scaler.save')

# Save the model
model.save('pose_model.h5')