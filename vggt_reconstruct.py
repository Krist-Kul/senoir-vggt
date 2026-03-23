"""VGGT 3D reconstruction: run inference on a folder of images."""

import os
import glob
import time
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Write ASCII PLY file."""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i].astype(np.uint8)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def reconstruct(image_folder: str, output_dir: str = "output", conf_threshold: float = 0.5) -> dict:
    """Run VGGT on all images in a folder. Returns dict with points, colors, point_maps, etc."""
    os.makedirs(output_dir, exist_ok=True)

    # Gather images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(image_folder, ext))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_folder}")
    print(f"[VGGT] Found {len(image_paths)} images in {image_folder}")

    # Load model
    print("[VGGT] Loading model...")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    model = VGGT.from_pretrained("facebook/VGGT")
    model = model.to(device)
    model.eval()
    print(f"[VGGT] Model loaded in {time.time()-t0:.1f}s (device={device}, amp={'on' if use_amp else 'off'})")

    # Preprocess images — keep float32, autocast handles mixed precision
    print("[VGGT] Preprocessing images...")
    images = load_and_preprocess_images(image_paths)  # (S, 3, H, W)
    images = images.unsqueeze(0).to(device)  # (1, S, 3, H, W)

    # Inference — model stays float32, autocast wraps ops that benefit from lower precision
    print("[VGGT] Running inference...")
    t0 = time.time()
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(images)
        else:
            preds = model(images)
    print(f"[VGGT] Inference done in {time.time()-t0:.1f}s")

    # Extract point maps: (1, S, H, W, 3) -> (S, H, W, 3)
    point_maps = preds["world_points"][0].float().cpu().numpy()
    S, H, W, _ = point_maps.shape

    # Extract confidence: (1, S, H, W) -> (S, H, W)
    conf = None
    if "world_points_conf" in preds:
        conf = preds["world_points_conf"][0].float().cpu().numpy()
    elif "depth_conf" in preds:
        conf = preds["depth_conf"][0].float().cpu().numpy()

    # Extract colors from preprocessed images: (1, S, 3, H, W) -> (S, H, W, 3)
    img_tensor = images[0].float().cpu().numpy()  # (S, 3, H, W)
    img_colors = np.transpose(img_tensor, (0, 2, 3, 1))  # (S, H, W, 3)
    img_colors = np.clip(img_colors * 255, 0, 255).astype(np.uint8)

    # Flatten to point cloud with confidence filtering
    all_points = []
    all_colors = []
    for s in range(S):
        pts = point_maps[s].reshape(-1, 3)
        cols = img_colors[s].reshape(-1, 3)

        if conf is not None:
            mask = conf[s].reshape(-1) > conf_threshold
            # Also filter invalid points (zeros or inf)
            valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).max(axis=1) < 1e6)
            mask = mask & valid
            pts = pts[mask]
            cols = cols[mask]
        else:
            valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).max(axis=1) < 1e6)
            pts = pts[valid]
            cols = cols[valid]

        all_points.append(pts)
        all_colors.append(cols)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    print(f"[VGGT] Point cloud: {len(points)} points (from {S} images, {H}x{W} resolution)")

    # Save raw PLY
    raw_ply = os.path.join(output_dir, "raw_pointcloud.ply")
    _write_ply(raw_ply, points, colors)
    print(f"[VGGT] Saved raw PLY: {raw_ply}")

    # Extract camera info
    cameras = {}
    if "pose_enc" in preds:
        cameras["pose_enc"] = preds["pose_enc"][0].float().cpu().numpy()
    if "depth" in preds:
        cameras["depth_maps"] = preds["depth"][0].float().cpu().numpy()

    return {
        "points": points,
        "colors": colors,
        "point_maps": point_maps,       # (S, H, W, 3) — needed for scale recovery
        "vggt_hw": (H, W),              # VGGT resolution
        "image_paths": image_paths,
        "cameras": cameras,
        "images": img_colors,            # (S, H, W, 3) uint8
        "conf": conf,                    # (S, H, W) or None
    }


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "leg_photos"
    result = reconstruct(folder)
    print(f"Done. {result['points'].shape[0]} points reconstructed.")
