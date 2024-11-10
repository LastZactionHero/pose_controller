import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import joblib
from tensorflow.keras.models import load_model
import pygame  # Import Pygame for joystick input

# Load the saved scaler and model
scaler = joblib.load('scaler.save')
model = load_model('pose_model.h5')

# Define the sequence length (should match the training script)
sequence_length = 3

# Initialize the pose buffer
pose_buffer = []

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

# Define the path to the model file (MediaPipe model)
model_path = "pose_landmarker.task"

# Download the model if it doesn’t exist locally
if not os.path.exists(model_path):
    import urllib.request
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    )
    urllib.request.urlretrieve(model_url, model_path)

# Initialize MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Open the camera stream
cap = cv2.VideoCapture(0)

# Get the number of features (pose landmarks)
# Assuming 33 landmarks with x, y, z coordinates
num_features = 33 * 3  # 99 features

# Initialize Pygame and Joystick
pygame.init()
pygame.joystick.init()

# Check if any joysticks are connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected. Please connect a joystick and try again.")
    # Optionally, you can exit or continue without joystick input
    joystick = None
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    num_axes = joystick.get_numaxes()
    num_buttons = joystick.get_numbuttons()
    num_hats = joystick.get_numhats()
    print(f"Joystick name: {joystick.get_name()}")
    print(f"Number of axes: {num_axes}")
    print(f"Number of buttons: {num_buttons}")
    print(f"Number of hats: {num_hats}")

# Deadzone threshold for controller axes
deadzone = 0.1

# Function to read controller inputs
def read_controller_inputs():
    pygame.event.pump()
    # Read axes
    axes = [joystick.get_axis(i) for i in range(num_axes)]
    # Read buttons
    buttons = [joystick.get_button(i) for i in range(num_buttons)]
    # Read hats (D-pad)
    hats = [joystick.get_hat(i) for i in range(num_hats)]
    return axes, buttons, hats

# Function to post controller state (stub)
def post_controller_state(controller_state):
    # Stub: In the future, this method will send the controller state to the server
    # For now, we just print it for debugging
    print("Controller State:")
    print(controller_state)

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
        # Add features to the pose buffer
        pose_buffer.append(features)
        if len(pose_buffer) > sequence_length:
            pose_buffer.pop(0)

        if len(pose_buffer) == sequence_length:
            # Prepare input for the model
            X_input = np.array(pose_buffer)
            # Reshape and scale
            X_input_scaled = scaler.transform(X_input.reshape(-1, num_features)).reshape(1, sequence_length, num_features)
            # Predict controller axes
            y_pred = model.predict(X_input_scaled)
            axes_pred = y_pred[0]

            # Read controller inputs
            if joystick:
                axes_ctrl, buttons, hats = read_controller_inputs()
            else:
                axes_ctrl = [0] * 6  # Default values if no joystick is connected
                buttons = []
                hats = []

            # Apply deadzone and override logic for axes 0 and 1
            # If controller axes 0 or 1 are outside deadzone, use controller values
            # Else, use pose predictions for axes 0 and 1
            axes_final = axes_ctrl.copy()  # Start with controller axes
            for i in [0, 1]:
                if abs(axes_ctrl[i]) > deadzone:
                    # Use controller axis value
                    axes_final[i] = axes_ctrl[i]
                else:
                    # Use pose prediction (ensure the predicted value is within [-1, 1])
                    axes_final[i] = np.clip(axes_pred[i], -1.0, 1.0)

            # Use controller values for axes 2-5
            # axes_final[2:] are already from controller

            # Build the controller state dictionary
            controller_state = {
                'axes': axes_final,
                'buttons': buttons,
                'hats': hats
            }

            # Post (print) the controller state
            post_controller_state(controller_state)

            # Display predictions on the frame
            for idx, axis_value in enumerate(axes_final):
                cv2.putText(
                    frame,
                    f"Axis {idx}: {axis_value:.3f}",
                    (10, 30 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # Optionally, display buttons and hats
            button_text = "Buttons: " + ' '.join([str(b) for b in buttons])
            cv2.putText(
                frame,
                button_text,
                (10, 30 + (len(axes_final) + 1) * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            hat_text = "Hats: " + ' '.join([str(h) for h in hats])
            cv2.putText(
                frame,
                hat_text,
                (10, 30 + (len(axes_final) + 2) * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Pose Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
if joystick:
    joystick.quit()
pygame.joystick.quit()
pygame.quit()
cv2.destroyAllWindows()
