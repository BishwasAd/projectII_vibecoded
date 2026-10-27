import os
import json
from ultralytics import YOLO
from tqdm import tqdm

def run_yolo_detections():
    print("🚀 Initializing Project Stella - Stage 2: YOLOv8 2D Detections...")
    
    frames_dir = "data/extracted_frames"
    output_json = "outputs/2d_detections.json"
    
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        print(f"❌ Error: No frames found in {frames_dir}")
        return

    # Load pretrained YOLOv8 model (Nano or Small for fast local inference)
    print("📥 Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    os.makedirs("outputs", exist_ok=True)
    detections_data = {}

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"🔍 Running detection across {len(frame_files)} frames...")

    for filename in tqdm(frame_files):
        img_path = os.path.join(frames_dir, filename)
        
        # Run YOLO inference
        results = model(img_path, verbose=False)
        
        frame_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates, confidence, and class name
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Filter low-confidence detections
                if conf > 0.4:
                    frame_detections.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "box_2d": xyxy
                    })
        
        if frame_detections:
            detections_data[filename] = frame_detections

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(detections_data, f, indent=4)

    print(f"✨ Saved 2D detections for {len(detections_data)} frames to {output_json}")

if __name__ == "__main__":
    run_yolo_detections()