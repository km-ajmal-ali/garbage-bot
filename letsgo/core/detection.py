"""
WasteBot Detection Module
==========================
Wraps the Hailo-accelerated YOLO inference from bounding_box.py
and provides higher-level helpers for the state machine.
"""

import sys
import os
import cv2
import numpy as np

from core.logger import get_logger

log = get_logger("detect")

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
from core.config import MODEL_PATH, CONFIDENCE_THRESHOLD, CAM_WIDTH, CAM_HEIGHT, CAMERA_ROTATE_180


# ───────────────────────────────────────────────────────────────────────────
# Model & Camera Factories
# ───────────────────────────────────────────────────────────────────────────

def load_model(hef_path: str = MODEL_PATH) -> HailoYOLOv8:
    """Load the YOLOv8 HEF model onto the Hailo AI HAT+."""
    log.info("Loading model: %s", hef_path)
    model = HailoYOLOv8(hef_path)
    log.info("Model loaded successfully. Input shape: %s", model.input_shape)
    return model


def init_camera(camera_index: int = 0,
                width: int = CAM_WIDTH,
                height: int = CAM_HEIGHT):
    """
    Open the best available camera backend.

    Returns:
        Camera object with .read() / .release() / .isOpened(),
        or None on failure.
    """
    log.info("Opening camera (index=%d, %dx%d) …", camera_index, width, height)
    cam = _open_camera(camera_index, width, height)
    if cam:
        log.info("Camera opened successfully")
    else:
        log.error("Failed to open camera on any backend!")
    return cam


def read_frame(camera):
    """
    Read a frame from the camera, applying 180° rotation if the
    camera is mounted upside-down (controlled by CAMERA_ROTATE_180).

    Returns:
        (ret, frame) – same contract as cv2.VideoCapture.read().
    """
    ret, frame = camera.read()
    if ret and frame is not None and CAMERA_ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return ret, frame


# ───────────────────────────────────────────────────────────────────────────
# Detection Helpers
# ───────────────────────────────────────────────────────────────────────────

def run_detection(model: HailoYOLOv8, frame: np.ndarray,
                  conf: float = CONFIDENCE_THRESHOLD,
                  camera=None) -> list:
    """
    Run YOLO inference on a single frame via the Hailo accelerator.

    Args:
        camera: Optional ThreadedCamera – paused during inference.

    Returns:
        List of detection dicts (see bounding_box.detect_objects).
    """
    detections = _detect_objects(model, frame, conf, camera=camera)
    log.debug("Inference complete: %d detection(s) (conf≥%.2f)", len(detections), conf)
    return detections


def estimate_object_depth(bbox_width_px: int) -> float:
    """
    Estimate object distance in centimetres using the pinhole model.

    Returns:
        Depth in cm, or -1.0 if not calculable.
    """
    depth = _estimate_depth(bbox_width_px)
    log.debug("Depth estimate: bbox_w=%dpx → %.1f cm", bbox_width_px, depth)
    return depth


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

    area = best['width_px'] * best['height_px']
    log.debug("Best target: '%s' conf=%.2f  bbox=%s  area=%dpx²  depth≈%.0fcm",
              best['label'], best['confidence'], best['bbox'], area,
              best.get('depth_cm', -1))
    return best
