import cv2
import dlib
import numpy as np
import os
from imutils import face_utils
from scipy.spatial import distance as dist

# Config
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
COUNTER = 0
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# EAR Function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


if not os.path.exists(PREDICTOR_PATH):
    print("Path not found!")
    exit()

print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Retrieve eye landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


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

    try:
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Slicing eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Compute average EAR
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            # For visual purpose
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    except Exception as e:
        print("Error:", e)
        continue

    cv2.imshow("Drowsiness Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()