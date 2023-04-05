import tensorflow as tf
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50

# Define the path to the root directory of your dataset
data_dir = 'asl_dataset'

# Define the path to the directory where you want to store the train and test data
output_dir = 'asl_split'

# Define the train-test split ratio
split_ratio = 0.2

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over each class directory in the root directory
for class_dir in os.listdir(data_dir):
    # Define the paths to the source and destination directories for this class
    src_dir = os.path.join(data_dir, class_dir)
    train_dir = os.path.join(output_dir, 'train', class_dir)
    test_dir = os.path.join(output_dir, 'test', class_dir)
    
    # Create the train and test directories for this class
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get the list of image files for this class
    image_files = os.listdir(src_dir)
    
    # Shuffle the image files
    random.shuffle(image_files)
    
    # Split the image files into train and test sets
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[split_index:]
    test_files = image_files[:split_index]
    
    # Copy the train files to the train directory
    for train_file in train_files:
        src_path = os.path.join(src_dir, train_file)
        dst_path = os.path.join(train_dir, train_file)
        shutil.copy(src_path, dst_path)
    
    # Copy the test files to the test directory
    for test_file in test_files:
        src_path = os.path.join(src_dir, test_file)
        dst_path = os.path.join(test_dir, test_file)
        shutil.copy(src_path, dst_path)
