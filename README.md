# Facial Emotion Recognition 🎭

This project is a real-time facial emotion recognition system. It uses a pre-trained model to predict emotions from facial expressions captured live from a webcam. The main features of this project include real-time face detection, emotion prediction, and live display of the results.

## ✨ Features

- **Real-time Face Detection**: The system uses OpenCV's Haar feature-based cascade classifiers to detect faces in real-time from a webcam feed.
- **Emotion Prediction**: The detected faces are then processed and fed into a pre-trained model which predicts the emotion expressed by the face.
- **Live Display**: The system displays the webcam feed with the detected faces and their predicted emotions in real-time.

## 🛠️ Technologies Used

- **Python**: The project is implemented in Python.
- **OpenCV**: Used for real-time face detection and image processing.
- **TensorFlow and Keras**: Used for loading the pre-trained model and predicting emotions.
- **NumPy**: Used for numerical operations on image data.

## 🚀 Installation Instructions

1. Clone the repository to your local machine.
2. Install the required libraries by running `pip install -r requirements.txt` in your terminal.
3. Run `python webcam.py` to start the facial emotion recognition system.

## 📚 Usage Examples

Here is an example of how to use the main feature of the project:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("FaceEmotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(preds)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow("Facial Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

This code will open your webcam, detect faces, predict their emotions, and display the results live. Press 'q' to quit.

## 📝 Note

The `Train_model.py` file is used to train the emotion recognition model. The trained model is saved as `FaceEmotion_model.h5`, which is then loaded in `webcam.py` for emotion prediction.