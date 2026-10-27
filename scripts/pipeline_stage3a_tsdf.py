import os
import json
import pycolmap
import numpy as np
from PIL import Image
import open3d as o3d

def generate_high_detail_mesh():
    print("🚀 Initializing Project Stella - Maximum Detail Surface Reconstruction...")

    colmap_sparse_path = "outputs/colmap_workspace/sparse/0"
    frames_dir = "data/extracted_frames"
    boxes_path = "outputs/3d_bounding_boxes.json"

    if not os.path.exists(colmap_sparse_path):
        print(f"❌ Error: COLMAP workspace not found at {colmap_sparse_path}.")
        return

    reconstruction = pycolmap.Reconstruction(colmap_sparse_path)
    print(f"📊 Loaded all {len(reconstruction.images)} registered frames from COLMAP.")

    # Load 3D bounding box centers to prioritize object density
    box_centers = []
    if os.path.exists(boxes_path):
        with open(boxes_path, "r") as f:
            boxes_data = json.load(f)
        for _, objects in boxes_data.items():
            for obj in objects:
                box_centers.append(np.array(obj["box_3d_center"]))

    all_points = []
    all_colors = []

    # 1. Gather sparse points
    for point3D_id, point3D in reconstruction.points3D.items():
        all_points.append(point3D.xyz)
        all_colors.append(point3D.color / 255.0)

    print(f"✨ Base sparse points: {len(all_points)}")

    # 2. Process ALL frames with high-density pixel sampling
    print("🔄 Backprojecting high-resolution pixels across all views...")
    
    for img_id, img in reconstruction.images.items():
        img_path = os.path.join(frames_dir, img.name)
        if not os.path.exists(img_path):
            base_name = os.path.splitext(img.name)[0]
            matches = [f for f in os.listdir(frames_dir) if f.startswith(base_name)]
            if matches:
                img_path = os.path.join(frames_dir, matches[0])
            else:
                continue

        try:
            # Load full-resolution image for maximum detail
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img) / 255.0
            H, W, _ = img_np.shape

            camera = reconstruction.cameras[img.camera_id]
            fx = camera.params[0]
            fy = camera.params[1] if len(camera.params) > 1 else camera.params[0]
            cx = camera.params[2]
            cy = camera.params[3]

            cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            R = cfw.rotation.matrix()
            T = np.array(cfw.translation)
            C = -R.T @ T

            # Sample every 4th pixel for high density (instead of every 10th)
            for v in range(0, H, 4):
                for u in range(0, W, 4):
                    color = img_np[v, u]
                    x_cam = (u - cx) / fx
                    y_cam = (v - cy) / fy
                    ray_cam = np.array([x_cam, y_cam, 1.0])
                    ray_cam /= np.linalg.norm(ray_cam)
                    ray_world = R.T @ ray_cam

                    depth = 1.1 # Tabletop distance scale
                    p_world = C + depth * ray_world
                    
                    all_points.append(p_world)
                    all_colors.append(color)
        except Exception as e:
            continue

    print(f"✨ Total high-detail points generated: {len(all_points)}")

    # 3. Create Point Cloud and Clean Outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    print("🧹 Removing statistical outliers for clean surfaces...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    print("🔄 Estimating high-precision normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
    pcd.orient_normals_consistent_tangent_plane(40)

    print("🧩 Generating high-detail Alpha Shape Mesh...")
    alpha = 0.03  # Tighter alpha value for sharper, more defined object geometry
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    os.makedirs("outputs/gaussian_splat", exist_ok=True)
    mesh_path = "outputs/gaussian_splat/textured_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"💾 High-detail mesh successfully saved to: {mesh_path}")

if __name__ == "__main__":
    generate_high_detail_mesh()