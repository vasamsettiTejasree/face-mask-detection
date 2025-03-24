#preprocessing of data
#importing libraries 
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

# Define dataset paths
with_mask_path = r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\images\face_withmask"
without_mask_path = r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\images\face_without_mask"

# Image settings
img_size = (128, 128)  # Resize all images to 128x128

X, y = [], []  # Lists to store image data and labels

# This function loads images, resizes them, normalizes pixel values, and assigns labels.
def load_images_from_folder(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip corrupted images
        img = cv2.resize(img, img_size)  # Resize image
        img = img / 255.0  # Normalize
        X.append(img)
        y.append(label)

# Load both datasets
load_images_from_folder(with_mask_path, label=0)  # 0 for 'With Mask'
load_images_from_folder(without_mask_path, label=1)  # 1 for 'Without Mask'

# Convert to NumPy arrays for easier processing.
X = np.array(X, dtype="float32")
y = np.array(y)

# Convert labels to categorical (for classification)
y = to_categorical(y, 2)

# Save preprocessed data
np.save("X.npy", X)   # Save images
np.save("y.npy", y)    # Save labels
# Print Dataset Information
print(f"Dataset Loaded: {len(X)} images")
print(f"Image Shape: {X.shape[1:]}") #Displays total number of images and their dimensions.

#training the model#

#importing libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#Sequential: Defines a linear model
#Cov2D: Convolutional layer for feature extraction.
#MaxPooling2D: Reduces image size while retaining features.
#Flatten: Converts 2D features into a 1D array.
#Dense: Fully connected layers for classification.
#Dropout: Reduces overfitting.
#Adam: Optimizer for training.

#train_test_split: Splits data into training and testing sets.

# Load previous preprocessed data
X = np.load("X.npy")
y = np.load("y.npy")

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#80% of data for training-20% of data for testing.


# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),  #First Conv Layer: 32 filters, 3×3 kernel, ReLU activation.
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),  #Second Conv Layer:64 filters
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes: With Mask (0), Without Mask (1)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)
#epochs=10: The model will train for 10 iterations.
# batch_size=16: Processes 16 images at a time
# Validation Data: Evaluates model performance after each epoch.

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Save model
model.save("face_mask_detector.h5")
print("Model saved as 'face_mask_detector.h5'")

#testing the model mask or no mask

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
try:
    model = load_model("face_mask_detector.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load OpenCV's DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\deploy.prototxt",
    r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\res10_300x300_ssd_iter_140000.caffemodel"
)
# Start webcam
video = cv2.VideoCapture(0) # 0 for default webcam
if not video.isOpened():
    print("Error: Could not access webcam.")
    exit()

while True:
    ret, frame = video.read() #Continuously reads frames from the webcam (video.read()).
    if not ret:
        print("Error: Failed to capture frame.")
        break

    h, w = frame.shape[:2]
    
    # Preprocess frame for face detection
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Ensure valid bounding box
            x, y, x1, y1 = max(0, x), max(0, y), min(w, x1), min(h, y1)

            # Extract face ROI
            face = frame[y:y1, x:x1]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue  # Skip invalid detections

            # Resize and normalize face
            face = cv2.resize(face, (128, 128))  # Resize to model input size
            face = face / 255.0  # Normalize
            face = np.expand_dims(face, axis=0)  # Add batch dimension

            # Predict mask or no mask
            prediction = model.predict(face)[0]
            label = "With Mask" if np.argmax(prediction) == 0 else "Without Mask"
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)

            # Display confidence score
            confidence_score = np.max(prediction) * 100
            label = f"{label} ({confidence_score:.2f}%)"

            # Draw rectangle and label on face
            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show output
    cv2.imshow("Face Mask Detector", frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
