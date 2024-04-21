import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the data
DATA_DIR = './data'

# Create data and labels lists to store hand landmark data and corresponding labels
data = []
labels = []

# Iterate through each directory (class) in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate through each image in the current directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Initialize a list to store hand landmark data for the current image
        data_aux = []
        
        # Lists to store x and y coordinates of landmarks
        x_ = []
        y_ = []

        # Read the image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the Hands model
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Iterate through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of each landmark and normalize them
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                
                # Normalize hand landmark coordinates relative to the minimum values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the normalized hand landmark data to the data list
            data.append(data_aux)
            # Append the label (directory name) to the labels list
            labels.append(dir_)

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)