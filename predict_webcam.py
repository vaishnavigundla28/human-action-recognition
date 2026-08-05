import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/human_action_recognition_model.keras")  # Ensure this file exists

# Define action labels
actions = ["sitting", "standing", "reading", "writing", "jumping", "walking"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Function to get keypoints
def get_keypoints(results):
    if results.pose_landmarks:
        keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(132)  # 33 keypoints * 4 values
    return keypoints

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    keypoints = get_keypoints(results)

    # Reshape and normalize keypoints
    keypoints = keypoints.reshape(1, 1, -1)
    keypoints = keypoints.reshape(1, 1, -1)
    prediction = model.predict(keypoints)

    # Predict action
    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction)

    # Display action
    action_text = f"Predicted: {actions[predicted_class]}"
    cv2.putText(frame, action_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Activity Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

