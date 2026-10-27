import os
import json
import open3d as o3d
import numpy as np

def visualize_clean_twin():
    print("🚀 Initializing Project Stella - Clean 3D Mesh Visualizer...")

    mesh_path = "outputs/gaussian_splat/textured_mesh.ply"
    boxes_path = "outputs/3d_bounding_boxes.json"

    geometries = []

    # 1. Load ONLY the solid textured mesh
    if os.path.exists(mesh_path):
        print(f"📂 Loading mesh from {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            geometries.append(mesh)
            print(f"✅ Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
        else:
            print("⚠️ Mesh file is empty.")
    else:
        print(f"❌ Error: Mesh not found at {mesh_path}.")

    # 2. Load 3D Bounding Boxes for reference
    centers = []
    if os.path.exists(boxes_path):
        with open(boxes_path, "r") as f:
            boxes_data = json.load(f)
        for _, objects in boxes_data.items():
            for obj in objects:
                center = obj["box_3d_center"]
                centers.append(center)
                obb = o3d.geometry.OrientedBoundingBox(center, np.eye(3), obj.get("extent", [0.4, 0.4, 0.6]))
                obb.color = (1.0, 0.8, 0.0) # Yellow boxes
                geometries.append(obb)

    if not geometries:
        print("❌ Error: No geometries to display!")
        return

    # 3. Launch Clean Visualizer Window
    print("✨ Launching Clean 3D Room Twin...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Project Stella - Clean Digital Twin", width=1280, height=720)

    for geom in geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.mesh_show_back_face = True
    opt.light_on = True  # Enable lighting for solid surface shading

    if centers:
        vis.poll_events()
        vis.update_renderer()
        vis.get_view_control().set_lookat(np.mean(centers, axis=0))

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    visualize_clean_twin()