import cv2
import numpy as np

video = cv2.VideoCapture("../data/videos/theft_test.mp4")

ret, prev_frame = video.read()

while True:

    ret, current_frame = video.read()

    if not ret:
        break

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, current_gray)

    motion_score = np.sum(diff) / 1000000

    print("Motion Score:", round(motion_score, 2))

    if motion_score > 15:
        print("⚠ Camera Shake Detected")

    prev_frame = current_frame

video.release()