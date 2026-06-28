"""Visualization utilities for point clouds and grasp poses."""

import numpy as np

from .rotation import rotation_6d_to_matrix


def create_hand_skeleton(
    translation: np.ndarray,
    rotation_6d: np.ndarray,
    joints: np.ndarray,
) -> dict:
    """Create a simplified hand skeleton representation for visualization.

    Args:
        translation: Wrist position, shape (3,).
        rotation_6d: 6D rotation, shape (6,).
        joints: Joint angles, shape (J,).

    Returns:
        Dictionary with 'wrist_pos', 'rotation_matrix', 'joint_angles'.
    """
    import torch

    rot_mat = rotation_6d_to_matrix(
        torch.from_numpy(rotation_6d).unsqueeze(0).float()
    ).squeeze(0).numpy()

    return {
        "wrist_pos": translation,
        "rotation_matrix": rot_mat,
        "joint_angles": joints,
    }


def visualize_grasp(
    xyz: np.ndarray,
    rgb: np.ndarray,
    left_pose: dict | None = None,
    right_pose: dict | None = None,
    save_path: str | None = None,
) -> None:
    """Visualize point cloud with grasp poses using Open3D.

    Args:
        xyz: Point positions, shape (N, 3).
        rgb: Point colors in [0, 1], shape (N, 3).
        left_pose: Left hand skeleton dict from create_hand_skeleton.
        right_pose: Right hand skeleton dict from create_hand_skeleton.
        save_path: If provided, save the visualization to this path.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Skipping visualization.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))

    geometries = [pcd]

    for pose, color in [
        (left_pose, [0.2, 0.6, 1.0]),
        (right_pose, [1.0, 0.4, 0.2]),
    ]:
        if pose is None:
            continue

        wrist = pose["wrist_pos"]
        rot = pose["rotation_matrix"]

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = wrist
        coord.transform(transform)
        geometries.append(coord)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(wrist)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)

        for i, axis in enumerate(rot.T):
            end = wrist + axis * 0.03
            points = np.stack([wrist, end])
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(points)
            lines.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_color = np.zeros(3)
            line_color[i] = 1.0
            lines.colors = o3d.utility.Vector3dVector([line_color])
            geometries.append(lines)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geometries:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        print(f"Saved visualization to {save_path}")
    else:
        o3d.visualization.draw_geometries(geometries, window_name="DexVLG Grasp")
