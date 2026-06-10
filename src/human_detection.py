import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("../data/videos/human_walk.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])

            if result.names[cls] == "person":
                print("Human Detected")

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()