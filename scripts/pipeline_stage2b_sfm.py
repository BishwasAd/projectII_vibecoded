import os
import pycolmap

def run_structure_from_motion(image_dir, output_colmap_dir):
    print("🚀 Initializing Project Stella - Stage 2B: Structure-from-Motion...")

    # 1. Ensure output directory exists
    os.makedirs(output_colmap_dir, exist_ok=True)
    
    database_path = os.path.join(output_colmap_dir, "database.db")
    if os.path.exists(database_path):
        os.remove(database_path)  # Clear old database if re-running

    print(f"📁 Image Directory: {image_dir}")
    print(f"💾 COLMAP Database Path: {database_path}")

    # 2. Extract SIFT features from extracted frames using PyCOLMAP v4 defaults
    print("🔍 Extracting SIFT features from frames...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir
    )

    # 3. Match features between image pairs exhaustively
    print("🔗 Matching features across frames (Exhaustive Matching)...")
    pycolmap.match_exhaustive(database_path=database_path)

    # 4. Run Incremental Mapping (Sparse SfM & Camera Poses)
    sparse_output_path = os.path.join(output_colmap_dir, "sparse")
    os.makedirs(sparse_output_path, exist_ok=True)

    print("📐 Running Incremental Structure-from-Motion (Calculating Camera Poses)...")
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_output_path
    )

    if len(maps) > 0:
        print(f"\n✅ Stage 2B Complete! Successfully generated {len(maps)} 3D reconstruction map(s).")
        print(f"📍 Sparse point cloud and camera poses saved to: {sparse_output_path}")
    else:
        print("\n❌ Reconstruction warning: No maps were generated. Check image overlap or texture.")

if __name__ == "__main__":
    image_input_dir = "data/extracted_frames"
    colmap_output_dir = "outputs/colmap_workspace"

    run_structure_from_motion(image_input_dir, colmap_output_dir)