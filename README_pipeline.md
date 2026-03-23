# 3D Leg Reconstruction Pipeline with Metric Scaling

Reconstruct a 3D point cloud of a leg from a few photos (5–20), scaled to real-world centimeters using an ASCM cube (ArUco-Supported Controlled Marker).

## How It Works

1. **Stage 1 — ArUco Detection** (`module_a.py`): Detects DICT_5X5_50 ArUco markers in each image, refines corners to sub-pixel accuracy, and picks the best detection as the scale reference.
2. **Stage 2 — VGGT Reconstruction** (`vggt_reconstruct.py`): Runs Facebook's [VGGT](https://huggingface.co/facebook/VGGT) model on all images to produce a dense 3D point cloud with colors.
3. **Stage 3 — Scale Recovery** (`scale_recovery.py`): Maps the ArUco corner pixels into VGGT's 3D point map, measures the marker's 3D edge length, and computes `scale_factor = 5.0 cm / mean_3d_edge`. The entire point cloud is then multiplied by this factor.

## Requirements

- Python 3.10+
- CUDA GPU (recommended, CPU fallback available)
- The VGGT model weights (auto-downloaded from HuggingFace on first run)

```bash
pip install -r requirements.txt
pip install -r requirements_pipeline.txt
```

## Usage

```bash
python pipeline.py --image_folder ./leg_photos --output_dir ./results
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--image_folder` | *(required)* | Folder containing input images (jpg/png/bmp/tiff) |
| `--output_dir` | `results` | Where to save outputs |
| `--marker_size_cm` | `5.0` | Known ArUco marker side length in cm |
| `--conf_threshold` | `0.5` | Minimum confidence for keeping 3D points |

### Input

- 5–20 photos of a leg with the ASCM cube visible in at least one image
- The cube uses ArUco DICT_5X5_50 markers (IDs 11–14) with a 5 cm side

### Output

```
results/
├── raw_pointcloud.ply      # Unscaled VGGT output
├── scaled_pointcloud.ply   # Metric-scaled point cloud (in cm)
├── scale_factor.json       # Scale metadata (factor, marker ID, confidence)
└── debug/
    └── *_aruco.jpg         # Annotated images showing detected markers
```

## Running Individual Modules

```bash
# Detect ArUco markers in a single image
python module_a.py path/to/image.jpg

# Run VGGT reconstruction only (no scaling)
python vggt_reconstruct.py path/to/image_folder
```

## File Overview

| File | Role |
|------|------|
| `module_a.py` | ArUco detection with sub-pixel refinement |
| `vggt_reconstruct.py` | VGGT inference, confidence filtering, PLY export |
| `scale_recovery.py` | 2D→3D corner mapping, metric scale computation |
| `pipeline.py` | CLI orchestrator |
| `requirements_pipeline.txt` | Additional dependencies for the pipeline |
