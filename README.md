# face-mask-detection

# Face Mask Detection and Recognition

## Introduction
This project focuses on detecting whether a person is wearing a mask and recognizing the person even when they are masked. The system utilizes deep learning techniques and OpenCV for real-time face detection and classification.

## Technologies Used
- Python
- OpenCV
- TensorFlow/Keras
- DeepFace
- NumPy
- SciPy
- dlib (for enhanced face recognition)
- Matplotlib (for visualization)
- Haar Cascade Classifier
- SSD (Single Shot MultiBox Detector)

## Installation
1. Install required dependencies:
   ```bash
   pip install opencv-python numpy tensorflow deepface scipy dlib matplotlib
   ```
2. Ensure you have the required pre-trained models:
   - `haarcascade_frontalface_default.xml`
   - `res10_300x300_ssd_iter_140000.caffemodel`
   - `shape_predictor_68_face_landmarks.dat` (for better face alignment)

## Dataset
The dataset consists of images with and without masks stored in the following directories:
- `images/face_withmask/`
- `images/face_without_mask/`

## Steps
1. **Data Collection:** Capture images with and without masks.
2. **Preprocessing:** Resize, normalize, and augment images for better model performance.
3. **Model Training:** Train a CNN model for mask detection and integrate DeepFace for face recognition.
4. **Face Recognition:** Use DeepFace and dlib for recognizing masked individuals with higher accuracy.
5. **Real-time Detection:** Utilize OpenCV to detect faces and predict mask usage and identity.
6. **Evaluation and Optimization:** Fine-tune model parameters and improve accuracy using hyperparameter tuning.

## Pretrained Models Used
Pretrained models help improve accuracy and reduce training time. Some key models used in this project include:
- **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`): Used for face detection.
- **SSD (Single Shot MultiBox Detector)** (`res10_300x300_ssd_iter_140000.caffemodel`): Provides more accurate face detection.
- **DeepFace Models**: VGG-Face, FaceNet, OpenFace, DeepID, Dlib, and ArcFace for face recognition.

## Applications
This system can be used in various real-world applications, including:
- **Security and Surveillance:** Identifying individuals in restricted areas while ensuring mask compliance.
- **Healthcare Monitoring:** Ensuring that staff and visitors wear masks in hospitals.
- **Public Spaces:** Automated monitoring in airports, malls, and offices.
- **Attendance Systems:** Masked face recognition for biometric authentication in schools and workplaces.
- **Smart Access Control:** Allowing access only to authorized individuals even if they are wearing masks.
- **Law Enforcement:** Helping police and security agencies identify suspects even with face coverings.

## Usage
Run the script to start real-time face mask detection and recognition:
```bash
python face_mask_detection.py
```

## Conclusion
This project integrates face detection, mask classification, and recognition to enhance security and health measures in public spaces. It demonstrates the effectiveness of deep learning in real-time applications and can be extended to work with more advanced biometric security systems.




