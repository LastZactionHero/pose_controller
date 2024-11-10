import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os
import time

# File to save pose data with labels
data_file = "pose_training_data.csv"
fieldnames = ["frame", "roll", "pitch", "label"] + [
    f"{part_name}_{axis}"
    for part_name in [mp.solutions.pose.PoseLandmark(i).name for i in range(33)]
    for axis in ["x", "y", "z"]
]

# Initialize data file
if not os.path.exists(data_file):
    with open(data_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

# Download the model if it doesn’t exist locally
model_url = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request

    urllib.request.urlretrieve(model_url, model_path)

# Define function to visualize landmarks
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

# Generate list of pose combinations
roll_values = [0, 30, 60, 90, 120, 150, 180]
pitch_values = [0, 30, 60, 90, 120, 150, 180]
pose_list = [(roll, pitch) for roll in roll_values for pitch in pitch_values]

# Create a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_count = 0

for roll, pitch in pose_list:
    label = f"ROLL_{roll}_PITCH_{pitch}"
    print(f"\nPrepare for pose: {label}")
    print("Press 's' to start recording.")

    # Wait for 's' key to start recording
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()
        cv2.putText(
            annotated_frame,
            f"Prepare for pose: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            annotated_frame,
            "Press 's' to start recording.",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Pose Data Collection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print("Starting in 2 seconds...")
            time.sleep(2)
            print("Recording started. Press 's' to stop.")
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Start recording pose data
    collected_data = []
    recording = True
    while recording:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        annotated_frame = draw_landmarks_on_image(frame, detection_result)
        cv2.putText(
            annotated_frame,
            f"Recording pose: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            "Press 's' to stop recording.",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Pose Data Collection", annotated_frame)

        # Collect pose data
        pose_landmarks_list = detection_result.pose_landmarks
        if pose_landmarks_list:
            data_row = {
                "frame": frame_count,
                "roll": roll,
                "pitch": pitch,
                "label": label,
            }
            pose_landmarks = pose_landmarks_list[0]  # Assuming single pose

            for i, landmark in enumerate(pose_landmarks):
                part_name = mp.solutions.pose.PoseLandmark(i).name
                data_row[f"{part_name}_x"] = round(landmark.x, 3)
                data_row[f"{part_name}_y"] = round(landmark.y, 3)
                data_row[f"{part_name}_z"] = round(landmark.z, 3)

            collected_data.append(data_row)
            frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print("Stopping recording...")
            recording = False
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Remove first and last 10 frames
    if len(collected_data) > 20:
        collected_data = collected_data[10:-10]
    else:
        print("Not enough data collected. Skipping this pose.")
        continue

    # Save collected data to CSV
    with open(data_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for data_row in collected_data:
            writer.writerow(data_row)

    print(f"Data for pose {label} saved. Proceeding to next pose.")

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()