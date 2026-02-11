import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input
import os

model = load_model("model/vgg19_face_model.h5")

dataset_path = "dataset"  # path to your dataset folder
class_names = sorted(os.listdir(dataset_path))  # list of subfolders (person names)
labels = {i: name for i, name in enumerate(class_names)}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        img = preprocess_input(np.expand_dims(face, axis=0))
        
        pred = model.predict(img)[0]
        class_id = np.argmax(pred)
        confidence = pred[class_id]
        label = labels[class_id]
        
        cv2.putText(frame, f"{label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Reconnaissance Faciale", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
