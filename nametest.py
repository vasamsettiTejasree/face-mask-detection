import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Paths for images (Masked & Unmasked)
known_faces = {
    "Tejasree": {
        "without_mask": r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\test_images\teju without maask.jpg",
        "with_mask": r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\test_images\teju withmaask.jpg"
    },
    "Hema": {
        "without_mask": r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\test_images\hema without mask.jpg",
        "with_mask": r"C:\Users\prath\Downloads\Face_Mask_Detection-main\Face_Mask_Detection-main\test_images\hema withmask.jpg"
    }
}

# Precompute embeddings for masked & unmasked faces
print("Encoding known faces...")
known_embeddings = {}
for name, paths in known_faces.items():
    embeddings = []
    for key, img_path in paths.items():
        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
            if embedding:
                embeddings.append(embedding[0]['embedding'])
        except:
            print(f"Could not process {img_path}")
    
    if embeddings:
        known_embeddings[name] = embeddings  # Store both embeddings

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
video = cv2.VideoCapture(0)
frame_count = 0  # Process every nth frame for speed

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 5 != 0:  # Skip frames to speed up processing
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Extract face region
        
        try:
            # Save detected face temporarily and get embedding
            cv2.imwrite("temp_face.jpg", face_img)
            detected_embedding = DeepFace.represent(img_path="temp_face.jpg", model_name="Facenet", enforce_detection=False)
            
            if detected_embedding:
                detected_embedding = detected_embedding[0]['embedding']
                recognized_name = "Unknown"
                
                # Compare embeddings using cosine similarity
                best_match_score = 0.5  # Cosine similarity threshold (Lower = better match)
                
                for name, embeddings in known_embeddings.items():
                    for known_embedding in embeddings:
                        similarity = 1 - cosine(known_embedding, detected_embedding)
                        if similarity > best_match_score:  # Higher similarity = better match
                            best_match_score = similarity
                            recognized_name = name
                
                # Display recognized name
                color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
                cv2.putText(frame, f"Person: {recognized_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        except Exception as e:
            print("Face not recognized properly:", e)
    
    cv2.imshow("Face Recognition with Mask", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
