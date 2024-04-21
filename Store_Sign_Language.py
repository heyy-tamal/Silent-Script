import os
import cv2

# Directory to store the collected data
DATA_DIR = './data'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (categories) to collect data for
number_of_classes = 1

no_class_exists = 0


# Number of images to collect for each class
dataset_size = 500

# Open the video capture device (webcam)
cap = cv2.VideoCapture(0)

# Loop through each class
for j in range(number_of_classes):
    # Create a directory for the current class
    if not os.path.exists(os.path.join(DATA_DIR, str(j+no_class_exists))):
        os.makedirs(os.path.join(DATA_DIR, str(j+no_class_exists)))

    print('Collecting data for class {}'.format(j+no_class_exists))

    # Wait for the user to press 'Q' to start collecting data for the current class
    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Counter to keep track of collected images
    counter = 0
    # Collect dataset_size number of images for the current class
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Save the current frame/image with a unique name in the appropriate class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j+no_class_exists), '{}.jpg'.format(counter)), frame)
        counter += 1

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()