import cv2
import os

video_path = "../data/videos/theft_test.mp4"
output_folder = "../data/frames"

os.makedirs(output_folder, exist_ok=True)

video = cv2.VideoCapture(video_path)

fps = int(video.get(cv2.CAP_PROP_FPS))

count = 0
saved = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    if count % (fps * 2) == 0:
        frame_name = os.path.join(
            output_folder,
            f"frame_{saved}.jpg"
        )

        cv2.imwrite(frame_name, frame)

        print(f"Saved: {frame_name}")

        saved += 1

    count += 1

video.release()

print("Done!")
print("Total Frames Saved:", saved)