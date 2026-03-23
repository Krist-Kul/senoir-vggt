"""Main pipeline: 3D leg reconstruction with metric scaling via ASCM cube."""

import argparse
import os
import glob
import time

from module_a import detect_ascm
from vggt_reconstruct import reconstruct, _write_ply
from scale_recovery import apply_metric_scale, save_scale_info


def find_images(folder: str) -> list:
    """Find all image files in folder."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    paths = sorted(p for ext in exts for p in glob.glob(os.path.join(folder, ext)))
    return paths


def main():
    parser = argparse.ArgumentParser(description="3D leg reconstruction with metric scaling")
    parser.add_argument("--image_folder", required=True, help="Folder with input images")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--marker_size_cm", type=float, default=5.0, help="ArUco marker side length in cm")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Point confidence threshold")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "debug")

    # ── Stage 1: ArUco detection on all images ──
    print("=" * 60)
    print("STAGE 1: ArUco marker detection")
    print("=" * 60)
    image_paths = find_images(args.image_folder)
    if not image_paths:
        print(f"ERROR: No images found in {args.image_folder}")
        return

    print(f"Scanning {len(image_paths)} images for ArUco markers...")
    detections = []
    for img_path in image_paths:
        print(f"  Processing: {os.path.basename(img_path)}")
        result = detect_ascm(img_path, known_size_cm=args.marker_size_cm, debug_dir=debug_dir)
        detections.append(result)
        if result["best"]:
            b = result["best"]
            print(f"    -> Found marker ID={b['marker_id']}, "
                  f"pixel_side={b['pixel_side']:.1f}px, conf={b['confidence']:.3f}")

    # Find best detection across all images
    best_idx = None
    best_conf = -1.0
    for i, det in enumerate(detections):
        if det["best"] and det["best"]["confidence"] > best_conf:
            best_conf = det["best"]["confidence"]
            best_idx = i

    if best_idx is None:
        print("\nERROR: No ArUco markers detected in any image. Cannot recover metric scale.")
        print("Ensure the ASCM cube is visible in at least one photo.")
        return

    best_det = detections[best_idx]
    best_image = image_paths[best_idx]
    print(f"\nBest marker detection: image #{best_idx} ({os.path.basename(best_image)})")
    print(f"  Marker ID={best_det['best']['marker_id']}, confidence={best_conf:.3f}")

    # ── Stage 2: VGGT reconstruction ──
    print("\n" + "=" * 60)
    print("STAGE 2: VGGT 3D reconstruction")
    print("=" * 60)
    vggt_result = reconstruct(args.image_folder, output_dir=args.output_dir,
                              conf_threshold=args.conf_threshold)

    # Find the index of the best marker image in VGGT's sorted image list
    marker_vggt_idx = vggt_result["image_paths"].index(best_image)
    print(f"Marker image index in VGGT: {marker_vggt_idx}")

    # ── Stage 3: Scale recovery ──
    print("\n" + "=" * 60)
    print("STAGE 3: Metric scale recovery")
    print("=" * 60)
    scaled_points, scale_factor = apply_metric_scale(
        ascm_result=best_det,
        vggt_result=vggt_result,
        marker_image_index=marker_vggt_idx,
        original_image_path=best_image,
        known_size_cm=args.marker_size_cm,
    )

    # ── Save outputs ──
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Scaled PLY
    scaled_ply = os.path.join(args.output_dir, "scaled_pointcloud.ply")
    _write_ply(scaled_ply, scaled_points, vggt_result["colors"])
    print(f"Scaled PLY: {scaled_ply} ({len(scaled_points)} points)")

    # Scale metadata
    save_scale_info(args.output_dir, scale_factor, best_det, len(scaled_points))

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Images processed: {len(image_paths)}")
    print(f"  Markers detected in: {sum(1 for d in detections if d['best'])} image(s)")
    print(f"  Scale reference: marker ID={best_det['best']['marker_id']} "
          f"({args.marker_size_cm} cm)")
    print(f"  Scale factor: {scale_factor:.4f} cm/unit")
    print(f"  Point cloud: {len(scaled_points)} points")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
