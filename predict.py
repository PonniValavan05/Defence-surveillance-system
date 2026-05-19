import cv2
import face_recognition
import os
import torch
from ultralytics import YOLO
from playsound import playsound
import yagmail
import time

# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = "techproembedded@gmail.com"
SENDER_PASS  = "wtufahsfxbqbltxh"
RECEIVER_EMAIL = "abninfotechprojects@gmail.com"

def send_email_alert(subject, body):
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASS)
        yag.send(RECEIVER_EMAIL, subject, body)
        print("📧 Email Sent Successfully")
    except Exception as e:
        print("Email Error:", e)

# ---------------- ALARM FUNCTION ----------------
def trigger_alarm():
    print("🔔 ALARM TRIGGERED")
    playsound("3.wav")

# ---------------- LOAD KNOWN FACES ----------------
def load_known_faces(known_faces_dir):
    encodings = []
    names = []
    for person in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person)
        for img in os.listdir(person_folder):
            path = os.path.join(person_folder, img)
            image = face_recognition.load_image_file(path)
            face_enc = face_recognition.face_encodings(image)
            if face_enc:
                encodings.append(face_enc[0])
                names.append(person)
    return encodings, names

known_encodings, known_names = load_known_faces("Known_Faces")

# ---------------- LOAD YOLO MODELS ----------------
weapon_model = YOLO("weap.pt")
fire_model   = YOLO("fire.pt")

DEVICE = 0 if torch.cuda.is_available() else "cpu"
weapon_model.to(DEVICE)
fire_model.to(DEVICE)

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)

last_alert_time = 0
alert_cooldown = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------- FACE RECOGNITION ----------------
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    person_detected = False
    intruder_detected = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
            color = (0,255,0)
        else:
            color = (0,0,255)
            intruder_detected = True

        person_detected = True

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, bottom+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ---------------- WEAPON DETECTION ----------------
    weapon_results = weapon_model.predict(frame, conf=0.5, device=DEVICE, verbose=False)
    weapon_boxes = weapon_results[0].boxes

    weapon_classes = ['gun','knife','pistol','rifle']
    weapon_found = False

    if weapon_boxes is not None:
        for box in weapon_boxes:
            cls = int(box.cls[0])
            label = weapon_model.names[cls]

            if label in weapon_classes:
                weapon_found = True
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    # ---------------- FIRE DETECTION ----------------
    fire_results = fire_model.predict(frame, conf=0.5, device=DEVICE, verbose=False)
    fire_boxes = fire_results[0].boxes

    fire_detected = False

    if fire_boxes is not None and len(fire_boxes)>0:
        fire_detected = True
        for box in fire_boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,"FIRE",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

    # ---------------- ALERT CONDITIONS ----------------
    current_time = time.time()

    # Intruder Condition
    if person_detected and weapon_found and intruder_detected:
        cv2.putText(frame,"⚠ INTRUDER DETECTED",
                    (50,50),cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,(0,0,255),3)

        if current_time - last_alert_time > alert_cooldown:
            trigger_alarm()
            send_email_alert("INTRUDER ALERT",
                             "Unknown person with weapon detected!")
            last_alert_time = current_time

    # Fire Condition
    if fire_detected:
        cv2.putText(frame,"🔥 FIRE ALERT",
                    (50,100),cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,(0,0,255),3)

        if current_time - last_alert_time > alert_cooldown:
            trigger_alarm()
            send_email_alert("FIRE ALERT",
                             "Fire detected in forest area!")
            last_alert_time = current_time

    cv2.imshow("FOREST SECURITY SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()