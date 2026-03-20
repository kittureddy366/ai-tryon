import cv2
from pose.body_pose import BodyPoseDetector
import mediapipe as mp

cap = cv2.VideoCapture(0)

pose_detector = BodyPoseDetector()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

timestamp = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    timestamp += 1

    result = pose_detector.detect_pose(frame, timestamp)

    if result.pose_landmarks:

        for pose_landmarks in result.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

    cv2.imshow("Pose Detection", frame)

    # PRESS Q TO EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows(),