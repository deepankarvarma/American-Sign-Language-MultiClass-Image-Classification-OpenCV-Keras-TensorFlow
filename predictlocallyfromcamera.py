import cv2
import numpy as np 
import tensorflow as tf

# Load the trained model
model=tf.keras.models.load_model('asl_image_classification.h5')

class_names = ['0', '1', '2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] # replace with your class names

def preprocess_image(image):
    # Resize the image to the input size of the model (224x224)
    resized_image = cv2.resize(image, (224, 224))
    # Convert the image to a format that can be used by the model (float32 array)
    input_image = resized_image.astype('float32') / 255.0
    # Add an extra dimension to the input to match the input shape of the model (batch size of 1)
    input_image = tf.expand_dims(input_image, axis=0)
    return input_image

cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Preprocess the frame
    input_image = preprocess_image(frame)

    # Make a prediction using the model
    prediction = model.predict(input_image)[0]

    # Get the predicted class and probability
    predicted_class = class_names[np.argmax(prediction)]
    predicted_prob = np.max(prediction)

    # Draw the predicted class and probability on the frame
    cv2.putText(frame, f'Predicted class: {predicted_class}, Probability: {predicted_prob}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # If the user presses the 'q' key, exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()