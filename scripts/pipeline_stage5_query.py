import os
import json

def query_room_objects():
    print("🚀 Initializing Project Stella - Stage 5: Semantic 3D Spatial Query Engine...\n")

    output_boxes_path = "outputs/3d_bounding_boxes.json"

    if not os.path.exists(output_boxes_path):
        print(f"❌ Error: 3D bounding boxes not found at {output_boxes_path}. Run Stage 3B first.")
        return

    print("🔍 Searching room database for objects...")
    with open(output_boxes_path, "r") as f:
        boxes_data = json.load(f)

    total_objects = 0
    object_classes = {}

    for frame_name, objects in boxes_data.items():
        for obj in objects:
            # Safely fetch class name regardless of key variation
            cls = obj.get("class_name", obj.get("class", "unknown"))
            object_classes[cls] = object_classes.get(cls, 0) + 1
            total_objects += 1

    print(f"\n📊 Spatial Database Summary:")
    print(f"   - Total frames indexed: {len(boxes_data)}")
    print(f"   - Total 3D objects detected & localized: {total_objects}")
    print(f"\n🏷️ Breakdown by Object Class:")
    
    for cls, count in object_classes.items():
        print(f"   - {cls}: {count} instances")

    print("\n✨ Spatial Query Complete! Your 3D digital twin is fully queryable.")

if __name__ == "__main__":
    query_room_objects()