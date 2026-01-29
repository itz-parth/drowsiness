import time
import cv2
import dlib
import numpy as np
import os
from imutils import face_utils
from scipy.spatial import distance as dist

# Config
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
MAR_AR_THRESH = 0.5
YAWN_CONSEC_FRAMES = 20

eye_counter = 0
yawn_counter = 0
yawn_history = []

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# EAR Function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR Function
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19]) 
    B = dist.euclidean(mouth[14], mouth[18]) 
    C = dist.euclidean(mouth[15], mouth[17]) 

    D = dist.euclidean(mouth[12], mouth[16])
    
    mar = (A + B + C) / (2.0 * D)
    return mar


if not os.path.exists(PREDICTOR_PATH):
    print("Path not found!")
    exit()

print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Retrieve eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


print("Starting video stream...")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Could not open camera.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    now = time.time()

    try:
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # Mouth coordinates
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            # For visual purpose
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # Eye detection
            if ear < EYE_AR_THRESH:
                eye_counter += 1
                if eye_counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                eye_counter = 0

            # Yawn detection
            if mar > MAR_AR_THRESH:
                yawn_counter += 1
            else:
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    yawn_history.append(now)
                yawn_counter = 0

            # Clean up old yawns
            yawn_history = [t for t in yawn_history if now - t < 60]

            if len(yawn_history) >= 3:
                cv2.putText(frame, "FREQUNET YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f} | YAWN_COUNTER: {yawn_counter} | YAWN_HISTORY: {len(yawn_history)}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    except Exception as e:
        print("Error:", e)
        continue

    cv2.imshow("Drowsiness Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()