"""Scale recovery: bridge Module A (ArUco) and VGGT to get metric point cloud."""

import os
import json
import numpy as np
import cv2


def _bilinear_sample_3x3(point_map: np.ndarray, x: float, y: float) -> np.ndarray:
    """Sample 3D point at (x, y) using median of 3x3 neighborhood for robustness."""
    H, W, _ = point_map.shape
    ix, iy = int(round(x)), int(round(y))
    samples = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            px, py = ix + dx, iy + dy
            if 0 <= px < W and 0 <= py < H:
                pt = point_map[py, px]
                if np.isfinite(pt).all() and np.abs(pt).max() < 1e6:
                    samples.append(pt)
    if not samples:
        # Fallback: single pixel
        iy_c = np.clip(iy, 0, H - 1)
        ix_c = np.clip(ix, 0, W - 1)
        return point_map[iy_c, ix_c]
    return np.median(samples, axis=0)


def apply_metric_scale(
    ascm_result: dict,
    vggt_result: dict,
    marker_image_index: int,
    original_image_path: str,
    known_size_cm: float = 5.0,
) -> tuple:
    """Compute metric scale from ArUco corners in VGGT point map.

    Returns (scaled_points Nx3, scale_factor).
    """
    best = ascm_result["best"]
    if best is None:
        raise ValueError("No ArUco marker detected — cannot recover scale.")

    corners_original = np.array(best["corners"])  # (4, 2) in original image coords

    # Get original image dimensions
    orig_img = cv2.imread(original_image_path)
    if orig_img is None:
        raise FileNotFoundError(f"Cannot read: {original_image_path}")
    orig_h, orig_w = orig_img.shape[:2]

    # VGGT resolution
    vggt_h, vggt_w = vggt_result["vggt_hw"]
    point_maps = vggt_result["point_maps"]  # (S, H, W, 3)
    point_map = point_maps[marker_image_index]  # (H, W, 3)

    # Map corners from original resolution to VGGT resolution
    scale_x = vggt_w / orig_w
    scale_y = vggt_h / orig_h
    corners_vggt = corners_original * np.array([scale_x, scale_y])

    print(f"[Scale] Original image: {orig_w}x{orig_h}, VGGT: {vggt_w}x{vggt_h}")
    print(f"[Scale] Marker ID={best['marker_id']}, pixel_side={best['pixel_side']:.1f}px")
    print(f"[Scale] Corner mapping scale: ({scale_x:.4f}, {scale_y:.4f})")

    # Look up 3D coordinates at each corner
    corners_3d = []
    for i, (cx, cy) in enumerate(corners_vggt):
        pt_3d = _bilinear_sample_3x3(point_map, cx, cy)
        corners_3d.append(pt_3d)
        print(f"  Corner {i}: pixel({cx:.1f}, {cy:.1f}) -> 3D({pt_3d[0]:.4f}, {pt_3d[1]:.4f}, {pt_3d[2]:.4f})")

    corners_3d = np.array(corners_3d)  # (4, 3)

    # Compute edge lengths between consecutive corners
    edge_lengths = []
    for i in range(4):
        p1 = corners_3d[i]
        p2 = corners_3d[(i + 1) % 4]
        dist = np.linalg.norm(p2 - p1)
        edge_lengths.append(dist)
        print(f"  Edge {i}->{(i+1)%4}: {dist:.6f} (VGGT units)")

    mean_edge_3d = np.mean(edge_lengths)
    if mean_edge_3d < 1e-10:
        raise ValueError(f"Mean 3D edge length is ~0 ({mean_edge_3d}). Bad reconstruction at marker location.")

    scale_factor = known_size_cm / mean_edge_3d
    print(f"[Scale] Mean 3D edge: {mean_edge_3d:.6f} VGGT units")
    print(f"[Scale] Scale factor: {scale_factor:.4f} (cm per VGGT unit)")

    # Apply scale to entire point cloud
    scaled_points = vggt_result["points"] * scale_factor

    # Sanity check: recompute marker edges in cm
    scaled_edges = [e * scale_factor for e in edge_lengths]
    print(f"[Scale] Verification — marker edges in cm: {[f'{e:.2f}' for e in scaled_edges]}")
    print(f"[Scale] Mean marker side: {np.mean(scaled_edges):.2f} cm (expected {known_size_cm:.1f} cm)")

    return scaled_points, float(scale_factor)


def save_scale_info(output_dir: str, scale_factor: float, ascm_result: dict, n_points: int):
    """Save scale metadata to JSON."""
    info = {
        "scale_factor": scale_factor,
        "known_size_cm": ascm_result["best"]["cm_side"],
        "marker_id": ascm_result["best"]["marker_id"],
        "marker_pixel_side": ascm_result["best"]["pixel_side"],
        "marker_confidence": ascm_result["best"]["confidence"],
        "source_image": ascm_result["image_path"],
        "n_scaled_points": n_points,
    }
    path = os.path.join(output_dir, "scale_factor.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[Scale] Saved scale info: {path}")


if __name__ == "__main__":
    print("This module is used by pipeline.py — not run standalone.")
