import os
import json
import pycolmap
import numpy as np

def lift_2d_to_3d():
    print("🚀 Initializing Project Stella - Stage 3B: 2D-to-3D Bounding Box Lifting...")

    colmap_sparse_path = "outputs/colmap_workspace/sparse/0"
    detections_path = "outputs/2d_detections.json"
    output_boxes_path = "outputs/3d_bounding_boxes.json"

    if not os.path.exists(colmap_sparse_path):
        print(f"❌ Error: COLMAP sparse reconstruction not found at {colmap_sparse_path}")
        return

    if not os.path.exists(detections_path):
        print(f"❌ Error: 2D detections not found at {detections_path}. Run Stage 2 first.")
        return

    print(f"📂 Loading COLMAP reconstruction from: {colmap_sparse_path}")
    reconstruction = pycolmap.Reconstruction(colmap_sparse_path)
    print(f"📊 Loaded COLMAP reconstruction with {len(reconstruction.images)} registered frames.")

    print(f"📂 Loading 2D detections from: {detections_path}")
    with open(detections_path, "r") as f:
        detections_data = json.load(f)

    colmap_images = {}
    for img_id, img in reconstruction.images.items():
        colmap_images[img.name] = img
        base_name = os.path.splitext(img.name)[0]
        colmap_images[base_name] = img

    lifted_boxes = {}
    matched_count = 0

    for frame_name, objects in detections_data.items():
        matched_img = None
        if frame_name in colmap_images:
            matched_img = colmap_images[frame_name]
        else:
            base_frame = os.path.splitext(frame_name)[0]
            if base_frame in colmap_images:
                matched_img = colmap_images[base_frame]

        if not matched_img:
            continue

        matched_count += 1
        img = matched_img
        camera = reconstruction.cameras[img.camera_id]

        frame_boxes = []
        for obj in objects:
            class_name = obj["class_name"]
            box_2d = obj["box_2d"]  # [x1, y1, x2, y2]
            confidence = obj["confidence"]

            u = (box_2d[0] + box_2d[2]) / 2.0
            v = (box_2d[1] + box_2d[3]) / 2.0

            estimated_depth = 2.0
            
            if camera.model.name in ["PINHOLE", "SIMPLE_PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"]:
                fx = camera.params[0]
                fy = camera.params[1] if len(camera.params) > 1 else fx
                cx = camera.params[2] if len(camera.params) > 2 else camera.width / 2.0
                cy = camera.params[3] if len(camera.params) > 3 else camera.height / 2.0

                X_c = (u - cx) * estimated_depth / fx
                Y_c = (v - cy) * estimated_depth / fy
                Z_c = estimated_depth

                # Universal PyCOLMAP rotation & translation extractor
                try:
                    if hasattr(img, "cam_from_world"):
                        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                        R = cfw.rotation.matrix()
                        t = np.array(cfw.translation)
                    elif hasattr(img, "rotation_matrix"):
                        R = img.rotation_matrix()
                        t = np.array(img.tvec)
                    else:
                        q = img.qvec
                        w, x, y, z = q[0], q[1], q[2], q[3]
                        R = np.array([
                            [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
                            [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
                            [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2]
                        ])
                        t = np.array(img.tvec)
                except Exception:
                    R = img.rotmat() if hasattr(img, "rotmat") else np.eye(3)
                    t = np.array(img.tvec)
                
                point_world = R.T @ (np.array([X_c, Y_c, Z_c]) - t)

                frame_boxes.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "box_3d_center": point_world.tolist(),
                    "extent": [0.5, 0.5, 0.8]
                })

        if frame_boxes:
            lifted_boxes[frame_name] = frame_boxes

    print(f"🔗 Successfully matched {matched_count} frames between detections and COLMAP.")

    os.makedirs("outputs", exist_ok=True)
    with open(output_boxes_path, "w") as f:
        json.dump(lifted_boxes, f, indent=4)

    print(f"✨ Successfully lifted detections into 3D! Saved results to {output_boxes_path}")

if __name__ == "__main__":
    lift_2d_to_3d()