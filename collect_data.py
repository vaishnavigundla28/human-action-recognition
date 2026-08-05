import cv2
import numpy as np
import mediapipe as mp
import time
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Define actions (Added "walking")
actions = ["sitting", "standing", "reading", "writing", "jumping", "walking"]
data_dir = "pose_data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def get_keypoints(results):
    """Extract keypoints from MediaPipe Pose results"""
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(132)  # 33 keypoints * 4 values (x, y, z, visibility)

def record_action(action_name, duration=10):
    """Record pose keypoints for a given action"""
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    keypoints_data = []

    print(f"Recording {action_name} for {duration} seconds...")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        keypoints = get_keypoints(results)
        keypoints_data.append(keypoints)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Recording: {action_name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    keypoints_data = np.array(keypoints_data)
    np.save(os.path.join(data_dir, f"{action_name}.npy"), keypoints_data)
    print(f"Saved {action_name}.npy with {keypoints_data.shape[0]} frames.")

if __name__ == "__main__":
    for action in actions:
        input(f"Press Enter to start recording {action}...")
        record_action(action)

