import os

# Path to the extracted train folder
dataset_path = r"C:\Users\tushar\Downloads\dogs-vs-cats\train\train"

# Show first 10 image names
print(os.listdir(dataset_path)[:10])

import cv2
import numpy as np

images = []
labels = []

# Load only first 100 cat images and first 100 dog images
for filename in os.listdir(dataset_path):
    if filename.startswith("cat.") and len([l for l in labels if l == 0]) < 100:
        img = cv2.imread(os.path.join(dataset_path, filename))
        img = cv2.resize(img, (64, 64))
        images.append(img.flatten())
        labels.append(0)

    elif filename.startswith("dog.") and len([l for l in labels if l == 1]) < 100:
        img = cv2.imread(os.path.join(dataset_path, filename))
        img = cv2.resize(img, (64, 64))
        images.append(img.flatten())
        labels.append(1)

print("Total Images Loaded:", len(images))
print("Total Labels:", len(labels))

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)