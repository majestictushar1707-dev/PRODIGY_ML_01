import os

# Food-101 dataset ka path yahan likho
dataset_path = r"C:\Users\tushar\Downloads\archive (2)\food-101\food-101\images"

# Dataset ke folders dikhaye
folders = os.listdir(dataset_path)

print("Total folders:", len(folders))
print("First 10 folders:", folders[:10])

import cv2
import numpy as np

images = []
labels = []

# Ignore hidden files/folders
food_classes = [f for f in folders if not f.startswith(".")]

# Sirf pehli 5 food classes use karenge
food_classes = food_classes[:5]

print("Selected Classes:", food_classes)

for food in food_classes:
    food_path = os.path.join(dataset_path, food)

    count = 0

    for image_name in os.listdir(food_path):
        image_path = os.path.join(food_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))

        images.append(img.flatten())
        labels.append(food)

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

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)