"""
WasteBot Detection Module
==========================
Wraps the Hailo-accelerated YOLO inference from bounding_box.py
and provides higher-level helpers for the state machine.
"""

import sys
import os
import numpy as np

# Ensure the project root is importable
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from test.bounding_box import (
    HailoYOLOv8,
    detect_objects   as _detect_objects,
    estimate_depth   as _estimate_depth,
    open_camera      as _open_camera,
    FOCAL_LENGTH_PX,
    KNOWN_OBJECT_WIDTH_CM,
)
from core.config import MODEL_PATH, CONFIDENCE_THRESHOLD, CAM_WIDTH, CAM_HEIGHT


# ───────────────────────────────────────────────────────────────────────────
# Model & Camera Factories
# ───────────────────────────────────────────────────────────────────────────

def load_model(hef_path: str = MODEL_PATH) -> HailoYOLOv8:
    """Load the YOLOv8 HEF model onto the Hailo AI HAT+."""
    print(f"[DETECT] Loading model: {hef_path}")
    return HailoYOLOv8(hef_path)


def init_camera(camera_index: int = 0,
                width: int = CAM_WIDTH,
                height: int = CAM_HEIGHT):
    """
    Open the best available camera backend.

    Returns:
        Camera object with .read() / .release() / .isOpened(),
        or None on failure.
    """
    print("[DETECT] Opening camera …")
    return _open_camera(camera_index, width, height)


# ───────────────────────────────────────────────────────────────────────────
# Detection Helpers
# ───────────────────────────────────────────────────────────────────────────

def run_detection(model: HailoYOLOv8, frame: np.ndarray,
                  conf: float = CONFIDENCE_THRESHOLD) -> list:
    """
    Run YOLO inference on a single frame via the Hailo accelerator.

    Returns:
        List of detection dicts (see bounding_box.detect_objects).
    """
    return _detect_objects(model, frame, conf)


def estimate_object_depth(bbox_width_px: int) -> float:
    """
    Estimate object distance in centimetres using the pinhole model.

    Returns:
        Depth in cm, or -1.0 if not calculable.
    """
    return _estimate_depth(bbox_width_px)


def pick_best_target(detections: list) -> dict | None:
    """
    Select the most prominent detection (largest bounding-box area).
    Attaches 'depth_cm' to the result if not already present.

    Returns:
        Detection dict, or None.
    """
    if not detections:
        return None

    best = max(detections,
               key=lambda d: d['width_px'] * d['height_px'])

    if 'depth_cm' not in best:
        best['depth_cm'] = estimate_object_depth(best['width_px'])

    return best
