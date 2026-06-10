import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import requests

# ==========================
# 🔐 TELEGRAM CONFIG
# ==========================

BOT_TOKEN = "8991539115:AAH55P4xU8fJvis9v1gse3xC8jPVIa4uImo"
CHAT_ID = "8786335732"

def send_telegram_alert(message, image_path=None):

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

        data = {
            "chat_id": CHAT_ID,
            "text": message
        }

        requests.post(url, data=data)

        if image_path:

            img_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

            with open(image_path, "rb") as img:
                requests.post(img_url, data={"chat_id": CHAT_ID}, files={"photo": img})

    except Exception as e:
        print("Telegram Error:", e)


# ==========================
# 📦 SETUP
# ==========================

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("❌ Camera Open Failed")
    exit()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📁 Output Folder:", OUTPUT_DIR)

alert_sent = False
shake_counter = 0

ret, prev_frame = video.read()

# ==========================
# 🎯 MAIN LOOP
# ==========================

while True:

    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # ==========================
    # 📊 MOTION DETECTION
    # ==========================

    diff = cv2.absdiff(prev_gray, gray)
    motion_score = np.sum(diff) / 1000000

    print(f"Motion Score: {motion_score:.2f}")

    # ==========================
    # 🔄 SHAKE DETECTION (3 frames)
    # ==========================

    if motion_score > 15:
        shake_counter += 1
    else:
        shake_counter = 0

    shake_detected = shake_counter >= 3

    # ==========================
    # 🤖 MOTION GATING (YOLO only if motion)
    # ==========================

    human_detected = False

    if motion_score > 5:

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])

                if r.names[cls] == "person":
                    human_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    cv2.putText(frame, "Human", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    else:
        print("⚡ No Motion → YOLO Skipped")

    # ==========================
    # 🚨 THEFT LOGIC
    # ==========================

    if human_detected and shake_detected and not alert_sent:

        alert_sent = True

        print("\n🚨 THEFT DETECTED 🚨")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_path = os.path.join(OUTPUT_DIR, f"theft_{timestamp}.jpg")

        cv2.imwrite(image_path, frame)

        print("📸 Image Saved:", image_path)

        send_telegram_alert(
            "🚨 THEFT DETECTED in SmartGuard AI!",
            image_path
        )

        log_path = os.path.join(OUTPUT_DIR, "theft_log.txt")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Theft Detected: {datetime.now()}\n")

        print("📝 Log Saved")

    elif human_detected:
        print("✅ Human Normal Activity")

    # ==========================
    # 🖥 DISPLAY
    # ==========================

    cv2.imshow("SmartGuard AI - Live", frame)

    prev_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================
# 🧹 CLEANUP
# ==========================

video.release()
cv2.destroyAllWindows()

print("✅ Program Ended")