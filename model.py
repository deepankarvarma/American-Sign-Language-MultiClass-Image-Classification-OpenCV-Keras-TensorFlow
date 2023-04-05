import tensorflow as tf
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model,Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
# Define the training and test data directories
train_dir = 'asl_split/train'
test_dir = 'asl_split/test'

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    seed =42)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  subset='validation',
                                                  shuffle=False,
                                                  seed=42)

# Define the MobileNetV2 model as the base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
)
base_model.trainable = False
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dropout(rate=0.2)(x)
outputs = Dense(units=5, activation="softmax")(x)

# Combine the base model with the new layers
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False
LEARNING_RATE = 0.001
# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=3,
                    validation_data=test_generator)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Saving the model
model.save('asl_image_classification.h5')