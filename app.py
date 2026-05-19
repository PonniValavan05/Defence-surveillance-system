import cv2
import os
import torch
import time
import face_recognition
from ultralytics import YOLO
from flask import Flask, render_template, redirect, url_for, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import yagmail
from playsound import playsound

# ---------------- FLASK CONFIG ----------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------------- DATABASE MODEL ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = "techproembedded@gmail.com"
SENDER_PASS  = "wtufahsfxbqbltxh"
RECEIVER_EMAIL = "abninfotechprojects@gmail.com"

def send_email(subject, body):
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASS)
        yag.send(RECEIVER_EMAIL, subject, body)
        print("Email Sent")
    except Exception as e:
        print("Email Error:", e)

def alarm():
    playsound("3.wav")

# ---------------- LOAD MODELS ----------------
weapon_model = YOLO("weap.pt")
fire_model = YOLO("fire.pt")

DEVICE = 0 if torch.cuda.is_available() else "cpu"
weapon_model.to(DEVICE)
fire_model.to(DEVICE)

# ---------------- LOAD FACES ----------------
def load_faces():
    encodings = []
    names = []
    for person in os.listdir("Known_Faces"):
        folder = os.path.join("Known_Faces", person)
        for img in os.listdir(folder):
            image = face_recognition.load_image_file(os.path.join(folder,img))
            enc = face_recognition.face_encodings(image)
            if enc:
                encodings.append(enc[0])
                names.append(person)
    return encodings, names

known_encodings, known_names = load_faces()

# ---------------- CAMERA GENERATOR ----------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    last_alert = 0
    cooldown = 10

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Recognition
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        person_detected = False
        intruder = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                color = (0,255,0)
            else:
                intruder = True
                color = (0,0,255)

            person_detected = True
            cv2.rectangle(frame,(left,top),(right,bottom),color,2)
            cv2.putText(frame,name,(left,bottom+20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

        # Weapon Detection
        weapon_results = weapon_model.predict(frame, conf=0.5, device=DEVICE, verbose=False)
        weapon_boxes = weapon_results[0].boxes
        weapon_found = False

        if weapon_boxes is not None:
            for box in weapon_boxes:
                cls = int(box.cls[0])
                label = weapon_model.names[cls]
                if label in ['gun','knife','pistol','rifle']:
                    weapon_found = True
                    x1,y1,x2,y2 = map(int,box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame,label,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        # Fire Detection
        fire_results = fire_model.predict(frame, conf=0.5, device=DEVICE, verbose=False)
        fire_boxes = fire_results[0].boxes
        fire_detected = False

        if fire_boxes is not None and len(fire_boxes)>0:
            fire_detected = True
            for box in fire_boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(frame,"FIRE",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

        current_time = time.time()

        if person_detected and weapon_found and intruder:
            cv2.putText(frame,"INTRUDER ALERT",(50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            if current_time - last_alert > cooldown:
                alarm()
                send_email("INTRUDER ALERT","Unknown person with weapon detected!")
                last_alert = current_time

        if fire_detected:
            cv2.putText(frame,"FIRE ALERT",(50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            if current_time - last_alert > cooldown:
                alarm()
                send_email("FIRE ALERT","Fire detected!")
                last_alert = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        user = User(username=request.form['username'],
                    password=request.form['password'])
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form['username'],
                                     password=request.form['password']).first()
        if user:
            login_user(user)
            return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/video")
@login_required
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)