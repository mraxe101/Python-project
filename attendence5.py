import cv2
import os
import csv
from datetime import datetime
import numpy as np


pics_folder = "D:/pics"  
attendance_folder = "D:/attendence5"  


if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("Error: Unable to load LBPHFaceRecognizer. Please install opencv-contrib-python.")
    exit()


def train_face_recognition_model():
    faces = []
    ids = []

 
    for image_name in os.listdir(pics_folder):
        image_path = os.path.join(pics_folder, image_name)
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces_detected) == 0:
                print(f"No face detected in image: {image_name}")
                continue

            for (x, y, w, h) in faces_detected:
                faces.append(gray[y:y+h, x:x+w]) 
                 
               
                person_id = int(image_name.split('.')[0])  
                ids.append(person_id)

   
    if len(faces) < 2:
        print("Error: Not enough face data for training. Please provide more images.")
        exit()

    
    recognizer.train(faces, np.array(ids))
    recognizer.save("face_trained.yml")
    print("Model trained successfully!")


def mark_attendance(person_id):
   
    current_date = datetime.now().strftime("%Y-%m-%d")
    attendance_file_path = os.path.join(attendance_folder, f"{current_date}.csv")

    
    if not os.path.isfile(attendance_file_path):
        with open(attendance_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Date', 'Time', 'Status']) 

   
    with open(attendance_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        time_now = datetime.now().strftime("%H:%M:%S")
        writer.writerow([person_id, current_date, time_now, "P"])  
    print(f"Attendance marked for ID: {person_id} as Present.")


def start_webcam_for_attendance():
    
    if not os.path.exists("face_trained.yml"):
        print("Training model, please wait...")
        train_face_recognition_model()
    else:
        recognizer.read("face_trained.yml")

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Starting face recognition...")

    
    recognized_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            
            if confidence < 100:
                name = "ID: {}".format(id_)  
            else:
                name = "Unknown"

            
            if id_ not in recognized_ids:
                print(f"New face detected with ID {id_}. Marking attendance...")
                mark_attendance(id_)  
                recognized_ids.add(id_)  

            confidence_text = f"Confidence: {round(100 - confidence)}%"
            cv2.putText(frame, f"ID: {id_}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, confidence_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        cv2.imshow('Face Attendance System', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam_for_attendance()
