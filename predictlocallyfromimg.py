import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('face_mask_classification.h5')

# Load the image and preprocess it
img = cv2.imread("images/354.jpg")
img = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img, axis=0)
img_array = img_array.astype('float32') / 255.

# Make a prediction using the model
prediction = model.predict(img_array)

# Get the predicted class and probability
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'] # replace with your class names
predicted_class = class_names[np.argmax(prediction)]
predicted_prob = np.max(prediction)

# Display the input image and predicted class

cv2.putText(img, f'{predicted_class},{predicted_prob}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
