import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# ==========================
# Setup
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output Folder:", OUTPUT_DIR)

# ==========================
# Load YOLO
# ==========================

model = YOLO("yolov8n.pt")

# ==========================
# Open Webcam
# ==========================

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Webcam Open Failed")
    exit()

print("✅ Webcam Opened Successfully")

ret, prev_frame = video.read()

if not ret:
    print("❌ First Frame Read Failed")
    exit()

# ==========================
# Variables
# ==========================

alert_sent = False
shake_counter = 0

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = video.read()

    if not ret:
        break

    human_detected = False

    # ----------------------
    # YOLO Human Detection
    # ----------------------

    results = model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            if result.names[cls] == "person":

                human_detected = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Human",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # ----------------------
    # Brightness Check
    # ----------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    black_screen = brightness < 40

    # ----------------------
    # Shake Detection
    # ----------------------

    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY
    )

    diff = cv2.absdiff(
        prev_gray,
        gray
    )

    motion_score = np.sum(diff) / 1000000

    if motion_score > 15:
        shake_counter += 1
    else:
        shake_counter = 0

    shake_detected = shake_counter >= 3

    print(
        f"Motion Score={motion_score:.2f} | Counter={shake_counter}"
    )

    # ----------------------
    # Final Theft Logic
    # ----------------------

    if human_detected and (black_screen or shake_detected):

        if not alert_sent:

            alert_sent = True

            print("\n🚨 POSSIBLE THEFT DETECTED 🚨")

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            image_path = os.path.join(
                OUTPUT_DIR,
                f"theft_{timestamp}.jpg"
            )

            saved = cv2.imwrite(
                image_path,
                frame
            )

            print("Image Saved:", saved)
            print("Saved To:", image_path)

            log_path = os.path.join(
                OUTPUT_DIR,
                "theft_log.txt"
            )

            with open(
                log_path,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"Theft Detected: {datetime.now()}\n"
                )

            print("Log Saved")

    # ----------------------
    # Display
    # ----------------------

    cv2.putText(
        frame,
        f"Shake Count: {shake_counter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Camera Theft Detection",
        frame
    )

    prev_frame = frame.copy()

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

# ==========================
# Cleanup
# ==========================

video.release()
cv2.destroyAllWindows()

print("\n✅ Program Finished")