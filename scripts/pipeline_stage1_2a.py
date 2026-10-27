import os
import cv2
import json
from ultralytics import YOLO

def process_video_pipeline(video_path, output_frame_dir, output_json_path):
    print("🚀 Initializing Project Stella - Stage 1 & Stage 2A Pipeline...")

    # 1. Ensure output directories exist
    os.makedirs(output_frame_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 2. Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video Loaded: {total_frames} total frames detected at {fps:.2f} FPS.")

    # 3. Load the YOLOv8 object detection model (runs automatically on your RTX 5060 GPU!)
    print("🤖 Loading YOLOv8 model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")  # 'n' stands for nano (fast & lightweight)

    frame_idx = 0
    saved_count = 0
    all_detections = {}

    # We skip every N frames to avoid capturing too many redundant, identical pictures
    frame_skip_interval = 5  

    print("🔄 Processing frames, checking blur, and running YOLOv8 detection...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every 5th frame
        if frame_idx % frame_skip_interval == 0:
            # --- BLUR DETECTION (Laplacian Variance) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Lowered threshold to 10.0 so it doesn't accidentally skip normal video frames
            if sharpness_score < 2.0:
                print(f"⚠️ Frame {frame_idx} skipped (Too blurry, score: {sharpness_score:.2f})")
                frame_idx += 1
                continue

            # Save the clean frame as a PNG image
            frame_filename = f"frame_{frame_idx:05d}.png"
            frame_save_path = os.path.join(output_frame_dir, frame_filename)
            cv2.imwrite(frame_save_path, frame)

            # --- YOLOv8 OBJECT DETECTION ---
            results = model(frame, verbose=False)
            frame_objects = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    frame_objects.append({
                        "class": cls_name,
                        "confidence": round(conf, 2),
                        "box_2d": [round(coord, 2) for coord in xyxy]
                    })

            all_detections[frame_filename] = {
                "sharpness_score": round(sharpness_score, 2),
                "detected_objects": frame_objects
            }

            saved_count += 1
            if saved_count % 5 == 0:
                print(f"✨ Processed and saved {saved_count} clean frames...")

        frame_idx += 1

    # Release video capture tool
    cap.release()

    # 4. Save all 2D bounding box detections to a JSON file
    with open(output_json_path, "w") as f:
        json.dump(all_detections, f, indent=4)

    print("\n✅ Stage 1 & Stage 2A Complete!")
    print(f"📁 Extracted clean frames saved to: {output_frame_dir}")
    print(f"📊 Semantic 2D detections saved to: {output_json_path}")

if __name__ == "__main__":
    video_file = "data/raw_videos/room.mp4"
    frame_output_dir = "data/extracted_frames/"
    json_output_file = "outputs/2d_detections.json"

    process_video_pipeline(video_file, frame_output_dir, json_output_file)