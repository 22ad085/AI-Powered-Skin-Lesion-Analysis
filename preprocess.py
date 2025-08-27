import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Define dataset paths
TRAIN_PATH = r"C:\Users\karth\OneDrive\Documents\Melanoma Project\dataset\melanoma_cancer_dataset\train"
TEST_PATH = r"C:\Users\karth\OneDrive\Documents\Melanoma Project\dataset\melanoma_cancer_dataset\test"

# Data Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True
)

# Only Rescaling for Test Set
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load Testing Data
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("✅ Data Preprocessing Done!")
