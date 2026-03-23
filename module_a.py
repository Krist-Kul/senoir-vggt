"""Module A: ArUco detection & scale extraction from ASCM cube images."""

import os
import numpy as np
import cv2


def detect_ascm(image_path: str, known_size_cm: float = 5.0, debug_dir: str = "debug") -> dict:
    """Detect ArUco markers (DICT_5X5_50) in an image and extract scale info.

    Returns dict with {marker_id: {corners, pixel_side, cm_side, confidence}, ..., 'best': {...}}
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return {"markers": {}, "best": None, "image_path": image_path}

    # Sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    for corners in corners_list:
        for i in range(4):
            pt = corners[0][i:i+1].astype(np.float32)
            refined = cv2.cornerSubPix(gray, pt, (5, 5), (-1, -1), criteria)
            corners[0][i] = refined[0]

    markers = {}
    best_id = None
    best_confidence = -1.0

    for i, marker_id in enumerate(ids.flatten()):
        corners = corners_list[i][0]  # shape (4, 2)
        perimeter = cv2.arcLength(corners.reshape(-1, 1, 2), closed=True)
        pixel_side = perimeter / 4.0

        # Confidence: how square-like the marker is (ratio of min/max side length)
        sides = []
        for j in range(4):
            p1 = corners[j]
            p2 = corners[(j + 1) % 4]
            sides.append(np.linalg.norm(p2 - p1))
        confidence = min(sides) / max(sides) if max(sides) > 0 else 0.0

        markers[int(marker_id)] = {
            "corners": corners.tolist(),
            "pixel_side": float(pixel_side),
            "cm_side": known_size_cm,
            "confidence": float(confidence),
        }

        if confidence > best_confidence:
            best_confidence = confidence
            best_id = int(marker_id)

    result = {
        "markers": markers,
        "best": {
            "marker_id": best_id,
            **markers[best_id],
        } if best_id is not None else None,
        "image_path": image_path,
    }

    # Save debug image
    os.makedirs(debug_dir, exist_ok=True)
    debug_img = image.copy()
    cv2.aruco.drawDetectedMarkers(debug_img, corners_list, ids)
    for i, marker_id in enumerate(ids.flatten()):
        corners = corners_list[i][0]
        for j, pt in enumerate(corners):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(debug_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(debug_img, f"C{j}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    debug_path = os.path.join(debug_dir, f"{basename}_aruco.jpg")
    cv2.imwrite(debug_path, debug_img)
    print(f"  [Module A] Saved debug image: {debug_path}")

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python module_a.py <image_path>")
        sys.exit(1)
    result = detect_ascm(sys.argv[1])
    if result["best"]:
        b = result["best"]
        print(f"Best marker ID={b['marker_id']}, pixel_side={b['pixel_side']:.1f}px, "
              f"confidence={b['confidence']:.3f}")
        print(f"Detected {len(result['markers'])} marker(s): {list(result['markers'].keys())}")
    else:
        print("No ArUco markers detected.")
