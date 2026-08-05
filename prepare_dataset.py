#loads each action file
import numpy as np
import os

# Define actions in the same order they were collected
actions = ["sitting", "standing", "reading", "writing", "jumping", "walking"]

X_data = []  # Features (pose keypoints)
y_data = []  # Labels (numerical representation of actions)

# Load each file and assign labels
for idx, action in enumerate(actions):
    file_path = f'pose_data/{action}.npy'
    
    if os.path.exists(file_path):
        data = np.load(file_path)  # Load keypoints for the action
        X_data.append(data)  # Add to feature list
        y_data.append(np.full((data.shape[0],), idx))  # Create corresponding labels

# Convert lists to numpy arrays
X_data = np.vstack(X_data)  # Stack vertically to form a dataset
y_data = np.concatenate(y_data)  # Merge labels into a single array

# Save the dataset
np.save("pose_data/X.npy", X_data)
np.save("pose_data/y.npy", y_data)

print("Dataset successfully created!")
print("X_data shape:", X_data.shape)  # Should be (total_samples, 132)
print("y_data shape:", y_data.shape)  # Should match total_samples

#combines all action data

#cell 24
import numpy as np
import os

# Define your actions
actions = ["sitting", "standing", "reading", "writing", "jumping", "walking"]

# Load the dataset
X_data = []
y_data = []

# Load each .npy file and assign correct labels
for idx, action in enumerate(actions):
    file_path = os.path.join("pose_data", f"{action}.npy")
    data = np.load(file_path)  # Load the .npy file
    X_data.append(data)  # Append pose keypoints
    y_data.append(np.full((data.shape[0],), idx))  # Assign the correct label

# Convert to numpy arrays
X_data = np.vstack(X_data)  # Stack all samples together
y_data = np.concatenate(y_data)  # Merge all labels

# Print the shapes
print("Shape of X_data:", X_data.shape)  # Should be (num_samples, 132)
print("Shape of y_data:", y_data.shape)  # Should be (num_samples,)


#reshapes the data for the LSTM

#cell 25
# Reshape for LSTM (add a time-step dimension)
X_data = X_data.reshape((X_data.shape[0], 1, X_data.shape[1]))

# Print the new shape
print("New shape of X_data:", X_data.shape)  # Should be (1689, 1, 132)

np.save("pose_data/X.npy", X_data)
np.save("pose_data/y.npy", y_data)

print("Dataset prepared and saved!")
