# Human Action Recognition using MediaPipe Pose Estimation and LSTM

## Overview

This project is a real-time human action recognition system that identifies human activities from webcam video using **MediaPipe Pose Estimation** and a **Long Short-Term Memory (LSTM)** neural network. The system extracts body pose keypoints from each video frame and predicts the performed action based on the sequence of detected poses.

The model is trained to recognize the following actions:

* Walking
* Sitting
* Standing
* Reading
* Writing
* Jumping

## Features

* Real-time action recognition using a webcam
* Pose keypoint extraction with MediaPipe
* LSTM-based sequence classification
* Modular project structure for data collection, preprocessing, training, and prediction
* Easily extendable to additional actions or datasets

## Technologies Used

* Python
* MediaPipe
* OpenCV
* TensorFlow / Keras
* NumPy
* Scikit-learn

## Project Structure

```text
human-action-recognition-mediapipe-lstm/
│                       
├── pose_data/                    # Processed dataset (X.npy, y.npy,Recorded pose sequence)
├── models/                       # Trained LSTM model
├── collect_data.py               # Data collection script
├── prepare_dataset.py            # Dataset preparation
├── train_model.py                # Model training
├── predict_webcam.py             # Real-time prediction                    
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Workflow

1. **Data Collection**
   Capture pose keypoints for each action using a webcam.

2. **Dataset Preparation**
   Convert recorded keypoints into training features and labels.

3. **Model Training**
   Train an LSTM neural network on the prepared pose sequences.

4. **Real-Time Prediction**
   Load the trained model and perform live action recognition through the webcam.

## Installation

Clone the repository:

```bash
git clone https://github.com/vaishnavigundla28/human-action-recognition.git
cd human-action-recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Collect Training Data

```bash
python collect_data.py
```

### Step 2: Prepare the Dataset

```bash
python prepare_dataset.py
```

### Step 3: Train the Model

```bash
python train_model.py
```

### Step 4: Run Real-Time Prediction

```bash
python predict_webcam.py
```

## Model

The project uses a **Long Short-Term Memory (LSTM)** neural network to learn temporal patterns in body pose sequences. MediaPipe Pose provides 3D body landmark coordinates, which are used as input features for action classification.

## Future Enhancements

* Support for additional human activities
* Higher prediction accuracy through larger datasets
* Sequence-based prediction using multiple consecutive frames
* Deployment as a web or desktop application
* Integration with attendance, surveillance, or fitness monitoring systems

## License

This project is intended for educational and research purposes.

## Author

**Vaishnavi Gundla**
Artificial Intelligence and Machine Learning (AIML)
