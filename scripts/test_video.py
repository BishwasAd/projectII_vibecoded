import os
import cv2

video_path = "data/raw_videos/room.mp4"

# Check if the file actually exists according to Python
print(f"Does file exist? {os.path.exists(video_path)}")
print(f"Absolute path being checked: {os.path.abspath(video_path)}")

# Try opening with OpenCV
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    print("✅ Success! OpenCV opened the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total Frames: {frames}")
else:
    print("❌ Failure: OpenCV could NOT open the video file. It might be an unsupported codec format (like HEVC/H.265).")

cap.release()