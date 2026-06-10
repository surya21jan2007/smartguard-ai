import cv2
import numpy as np

video = cv2.VideoCapture("../data/videos/theft_test.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    print("Brightness:", brightness)

    if brightness < 40:
        print("⚠ Possible Theft - Camera Covered")

    cv2.imshow("Theft Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()