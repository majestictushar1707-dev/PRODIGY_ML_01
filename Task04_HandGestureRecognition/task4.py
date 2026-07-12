import os

# Path to the extracted LeapGestRecog dataset
dataset_path = r"C:\Users\tushar\Downloads\archive (1)\leapGestRecog\leapGestRecog"

# Get all subject folders
folders = os.listdir(dataset_path)

# Display information
print("Total folders:", len(folders))
print("Folders:", folders)

import cv2
import numpy as np

images = []
labels = []

# Load only 100 images from each gesture
for person in folders:
    person_path = os.path.join(dataset_path, person)

    for gesture in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture)

        count = 0

        for image_name in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            images.append(img.flatten())
            labels.append(gesture)

            count += 1

            if count == 100:
                break

print("Total Images Loaded:", len(images))
print("Total Labels:", len(labels))

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)