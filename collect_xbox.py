import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
import pygame
import time

# Initialize Pygame and Joystick
pygame.init()
pygame.joystick.init()

# Check if any joysticks are connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected. Please connect a joystick and try again.")
    exit()
else:
    print(f"Number of joysticks detected: {joystick_count}")

# Get the first joystick (adjust the index if necessary)
joystick = pygame.joystick.Joystick(0)
joystick.init()

num_axes = joystick.get_numaxes()
num_buttons = joystick.get_numbuttons()
num_hats = joystick.get_numhats()

print(f"Joystick name: {joystick.get_name()}")
print(f"Number of axes: {num_axes}")
print(f"Number of buttons: {num_buttons}")
print(f"Number of hats: {num_hats}")

# File to save pose data with controller axes values
data_file = "pose_controller_data.csv"

# Create fieldnames for CSV
pose_fieldnames = [
    f"{part_name}_{axis}"
    for part_name in [mp.solutions.pose.PoseLandmark(i).name for i in range(33)]
    for axis in ["x", "y", "z"]
]
controller_fieldnames = [f"axis_{i}" for i in range(num_axes)]
fieldnames = pose_fieldnames + controller_fieldnames

# Initialize data file
if not os.path.exists(data_file):
    with open(data_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)

# Download the model if it doesn’t exist locally
model_url = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request
    urllib.request.urlretrieve(model_url, model_path)

# Initialize MediaPipe PoseLandmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# Function to visualize landmarks
def draw_landmarks_on_image(bgr_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(bgr_image)
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image

# Open the camera stream
cap = cv2.VideoCapture(0)
frame_count = 0

print("Press 's' to start recording. Press 'q' to quit.")

recording = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    annotated_frame = draw_landmarks_on_image(frame, detection_result)
    if recording:
        cv2.putText(
            annotated_frame,
            "Recording...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            annotated_frame,
            "Press 's' to start recording.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
    cv2.imshow("Pose Data Collection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        recording = not recording
        if recording:
            print("Recording started.")
        else:
            print("Recording stopped.")
    elif key == ord("q"):
        break

    if recording and detection_result.pose_landmarks:
        # Process event queue for controller input
        pygame.event.pump()

        # Collect pose data
        pose_landmarks = detection_result.pose_landmarks[0]
        data_row = []
        # Pose landmarks
        for landmark in pose_landmarks:
            data_row.extend([round(landmark.x, 3), round(landmark.y, 3), round(landmark.z, 3)])
        # Controller axes
        axes = [joystick.get_axis(i) for i in range(num_axes)]
        data_row.extend(axes)
        # Save data row to CSV
        with open(data_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

        frame_count += 1

cap.release()
joystick.quit()
pygame.joystick.quit()
pygame.quit()
cv2.destroyAllWindows()
