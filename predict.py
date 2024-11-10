import vgamepad as vg
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import joblib
from tensorflow.keras.models import load_model

# Load the saved scalers and model
scaler = joblib.load('scaler.save')
label_scaler = joblib.load('label_scaler.save')
model = load_model('pose_model.h5')

gamepad = vg.VX360Gamepad()

# Function to extract features from detection_result
def extract_features(detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    if not pose_landmarks_list:
        return None
    pose_landmarks = pose_landmarks_list[0]
    features = []
    for landmark in pose_landmarks:
        features.extend([landmark.x, landmark.y, landmark.z])
    return features

# Initialize MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Open the camera stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Extract features
    features = extract_features(detection_result)
    if features:
        # Normalize features
        features = scaler.transform([features])

        # Predict roll and pitch
        y_pred = model.predict(features)
        # Inverse transform to get original scale
        roll_pred, pitch_pred = label_scaler.inverse_transform(y_pred)[0]

        # Display predictions
        cv2.putText(
            frame,
            f"Roll: {roll_pred:.1f}°, Pitch: {pitch_pred:.1f}°",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        roll_input = (roll_pred - 90) / 90  # Maps 0-180 to -1 to 1
        pitch_input = (pitch_pred - 90) / 90  # Maps 0-180 to -1 to 1

        # Apply deadzone thresholds if needed
        deadzone = 0.1
        if abs(roll_input) < deadzone:
            roll_input = 0
        if abs(pitch_input) < deadzone:
            pitch_input = 0

        # Set the right thumbstick values (or any control you prefer)
        gamepad.right_joystick_float(x_value_float=roll_input, y_value_float=pitch_input)

        # Send updates to the virtual gamepad
        gamepad.update()

    cv2.imshow("Pose Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
