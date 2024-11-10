
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os

# Download the model if it doesn’t exist locally
model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request
    urllib.request.urlretrieve(model_url, model_path)

# Define function to visualize landmarks
def draw_landmarks_on_image(bgr_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(bgr_image)

    # Loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

# Print pose landmarks to console
def print_pose_landmarks(detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    if not pose_landmarks_list:
        return

    # Loop through the detected poses and landmarks
    for idx, pose_landmarks in enumerate(pose_landmarks_list):
        print(f"\nPose #{idx + 1} landmarks:")
        for i, landmark in enumerate(pose_landmarks):
            part_name = mp.solutions.pose.PoseLandmark(i).name
            print(f"{part_name}: (x: {landmark.x:.2f}, y: {landmark.y:.2f}, z: {landmark.z:.2f}, visibility: {landmark.visibility:.2f})")

# Create a PoseLandmarker object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# Open the camera stream
cap = cv2.VideoCapture(0)  # Adjust camera index as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run pose detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Print pose landmark data to the console
    print_pose_landmarks(detection_result)

    # Draw landmarks on the frame
    annotated_frame = draw_landmarks_on_image(frame, detection_result)

    # Display the annotated frame
    cv2.imshow("Pose Landmarks - Razer Kiyo", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()