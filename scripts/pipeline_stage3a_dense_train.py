import os
import json
import torch
import pycolmap
import numpy as np
from PIL import Image
import open3d as o3d

def generate_smooth_mesh():
    print("🚀 Initializing Project Stella - Stage 3A: Poisson Surface & Mesh Reconstruction...")

    colmap_sparse_path = "outputs/colmap_workspace/sparse/0"
    frames_dir = "data/extracted_frames"
    boxes_path = "outputs/3d_bounding_boxes.json"

    if not os.path.exists(colmap_sparse_path):
        print(f"❌ Error: COLMAP workspace not found at {colmap_sparse_path}.")
        return

    reconstruction = pycolmap.Reconstruction(colmap_sparse_path)
    print(f"📊 Loaded {len(reconstruction.images)} registered images from COLMAP.")

    all_points = []
    all_colors = []

    # 1. Collect COLMAP sparse points
    for point3D_id, point3D in reconstruction.points3D.items():
        all_points.append(point3D.xyz)
        all_colors.append(point3D.color / 255.0)

    # 2. Backproject keyframe pixels for surface density
    count = 0
    for img_id, img in reconstruction.images.items():
        img_path = os.path.join(frames_dir, img.name)
        if not os.path.exists(img_path):
            base_name = os.path.splitext(img.name)[0]
            matches = [f for f in os.listdir(frames_dir) if f.startswith(base_name)]
            if matches:
                img_path = os.path.join(frames_dir, matches[0])
            else:
                continue

        pil_img = Image.open(img_path).convert("RGB").resize((640, 360))
        img_np = np.array(pil_img) / 255.0
        H, W, _ = img_np.shape

        camera = reconstruction.cameras[img.camera_id]
        scale_x = W / camera.width
        scale_y = H / camera.height
        fx = camera.params[0] * scale_x
        fy = (camera.params[1] if len(camera.params) > 1 else camera.params[0]) * scale_y
        cx = camera.params[2] * scale_x
        cy = camera.params[3] * scale_y

        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R = cfw.rotation.matrix()
        T = np.array(cfw.translation)
        C = -R.T @ T

        # Sample pixels
        for v in range(0, H, 12):
            for u in range(0, W, 12):
                color = img_np[v, u]
                x_cam = (u - cx) / fx
                y_cam = (v - cy) / fy
                ray_cam = np.array([x_cam, y_cam, 1.0])
                ray_cam /= np.linalg.norm(ray_cam)
                ray_world = R.T @ ray_cam

                depth = 1.1 # Tabletop distance scale
                all_points.append(C + depth * ray_world)
                all_colors.append(color)

        count += 1
        if count >= 35:
            break

    # 3. Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    print("🔄 Estimating surface normals for mesh reconstruction...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    print("🧩 Generating smooth surface mesh via Poisson Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # Remove low-density outlier geometry to keep the room clean
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(densities < density_threshold)

    os.makedirs("outputs/gaussian_splat", exist_ok=True)
    mesh_path = "outputs/gaussian_splat/textured_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"💾 Smooth 3D mesh successfully saved to: {mesh_path}")

if __name__ == "__main__":
    generate_smooth_mesh()