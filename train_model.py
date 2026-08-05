import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

# Define the model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, 132)),  # First LSTM layer
    LSTM(64, return_sequences=False),  # Second LSTM layer
    Dense(32, activation='relu'),  # Fully connected layer
    Dense(len(actions), activation='softmax')  # Output layer (6 actions)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

from tensorflow.keras.optimizers import Adam

# Define a better LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, 132)),
    LSTM(128, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

# Compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with more epochs
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.save("models/human_action_recognition_model.keras")
print("Model saved successfully!")
