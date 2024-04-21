import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from the pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Extract data and labels from the dictionary
data = data_dict['data']
labels = data_dict['labels']

# Find the maximum length of a data sample
max_length = max(len(sample) for sample in data)

# Pad or truncate each data sample to the maximum length
for i in range(len(data)):
    data[i] = np.pad(data[i], (0, max_length - len(data[i])), mode='constant')

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a Random Forest classifier
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f) 