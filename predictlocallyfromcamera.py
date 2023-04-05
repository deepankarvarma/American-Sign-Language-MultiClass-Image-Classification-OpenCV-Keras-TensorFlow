import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('asl_image_classification.h5')

# Load the SSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Define the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()

    # Get image dimensions
    height, width, channels = frame.shape

    # Create a blob from the input image for SSD input
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input for SSD and run inference
    net.setInput(blob)
    detections = net.forward()

    # Get the bounding box coordinates and confidence scores
    conf_threshold = 0.5
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            box = box.astype(int)
            boxes.append(box)

    # Select the box with the highest confidence score
    if len(boxes) > 0:
        box = boxes[0]

        # Check that box dimensions are not zero
        if box[2] > 0 and box[3] > 0:
            # Crop the image to the hand region
            x, y, w, h = box
            hand_frame = frame[y:y+h, x:x+w]

            # Preprocess the hand image
            hand_frame = cv2.resize(hand_frame, (224, 224))
            hand_array = np.expand_dims(hand_frame, axis=0)
            hand_array = hand_array.astype('float32') / 255.

            # Make a prediction using the model
            prediction = model.predict(hand_array)

            # Get the predicted class and probability
            class_names = ['0', '1', '2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] # replace with your class names
            predicted_class = class_names[np.argmax(prediction)]
            predicted_prob = np.max(prediction)

            # Display the input image and predicted class
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{predicted_class},{predicted_prob}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('Input Image', frame)
            cv2.waitKey(1) 
        else:
            print("No hand detected in the image")
    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()
