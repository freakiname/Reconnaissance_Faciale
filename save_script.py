import cv2
import os
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python save_script.py NomPersonne")
        return

    person_name = sys.argv[1]
    save_path = f"dataset/{person_name}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    print(f"Capture des images pour {person_name}...")

    while count < 60:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            cv2.imwrite(f"{save_path}/{count}.jpg", face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/60", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Capture terminée.")

if __name__ == "__main__":
    main()
