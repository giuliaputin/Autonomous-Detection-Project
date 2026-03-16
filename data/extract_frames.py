import cv2
import os

video_path = "smiley_video.mp4"
output_folder = "data/train/smiley"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save every 5th frame
    if frame_count % 5 == 0:
        filename = f"smiley_{saved:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), frame)
        saved += 1

    frame_count += 1

cap.release()

print("Saved images:", saved)