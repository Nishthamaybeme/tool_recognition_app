import os
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Function to check disk usage of a given path
def check_disk_usage(path):
    total, used, free = shutil.disk_usage(path)
    return f"Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB"

# Disk usage before training
print("Disk Usage Before Training:")
print("C Drive:", check_disk_usage("C:/"))
print("E Drive:", check_disk_usage("E:/"))
# print("F Drive:", check_disk_usage("F:/"))

# Directories for data
train_data_dir = r'E:\tensorflow website\lenet architecture\train_data\train_data'

validation_data_dir = r'E:\tensorflow website\lenet architecture\validation_data_V2\validation_data_V2'

# Data generators for loading images
datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load VGG16 pre-trained model without the top layers
vgg_base = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

# Freeze the convolutional base layers so they are not trained
for layer in vgg_base.layers:
    layer.trainable = False

# Build the final model
vgg_model = models.Sequential([
    vgg_base,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(8, activation='softmax')  # Adjust the number of classes based on your dataset
])

# Compile the model
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model with cooldown between epochs
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    history = vgg_model.fit(
        train_generator,
        epochs=1,  # Train one epoch at a time
        validation_data=validation_generator
    )
    
    print(f"Epoch {epoch + 1}/{num_epochs} completed. Cooling down for 10 seconds...")
    time.sleep(10)  # Pause for 10 seconds (adjust as needed)
    print(f"Cooldown complete. Proceeding to epoch {epoch + 2}/{num_epochs}...\n" if epoch + 1 < num_epochs else "Training complete.")

# Disk usage after training
print("\nDisk Usage After Training:")
print("C Drive:", check_disk_usage("C:/"))
print("E Drive:", check_disk_usage("E:/"))
# print("F Drive:", check_disk_usage("F:/"))

# Save the model
try:
    vgg_model.save('E:/tensorflow website/streamlit/my_model.h5')
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {e}")

