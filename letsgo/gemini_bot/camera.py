"""
GeminiBot Camera Module
========================
Initialises the camera (Picamera2 preferred, OpenCV fallback)
and provides a unified read_frame() helper.
"""

import cv2
import numpy as np

from core.logger import get_logger
from gemini_bot.config import CAM_WIDTH, CAM_HEIGHT, CAMERA_ROTATE_180

log = get_logger("camera")


# ─────────────────────────────────────────────────────────────────────
# Camera init (Picamera2 → OpenCV fallback)
# ─────────────────────────────────────────────────────────────────────

def init_camera(index: int = 0,
                width: int = CAM_WIDTH,
                height: int = CAM_HEIGHT):
    """
    Open the best available camera backend.
    Tries Picamera2 first (native Pi CSI), falls back to OpenCV.

    Returns:
        Camera-like object with .read() and .release(), or None.
    """

    # ── Picamera2 (preferred on Raspberry Pi) ────────────────────────
    try:
        from picamera2 import Picamera2

        log.info("Trying Picamera2 backend …")
        picam = Picamera2()
        config = picam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        log.info("Picamera2 opened (%dx%d)", width, height)

        # Wrap in an adapter so we have .read() / .release()
        return _Picamera2Wrapper(picam)

    except Exception as e:
        log.warning("Picamera2 not available: %s — trying OpenCV", e)

    # ── OpenCV fallback ──────────────────────────────────────────────
    for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            log.info("OpenCV camera opened (backend=%d, %dx%d)",
                     backend, width, height)
            return cap
        cap.release()

    log.error("Could not open any camera backend!")
    return None


class _Picamera2Wrapper:
    """Thin wrapper that gives Picamera2 a cv2.VideoCapture-like API."""

    def __init__(self, picam):
        self._picam = picam

    def read(self):
        try:
            frame = self._picam.capture_array()
            # Picamera2 returns RGB by default; convert to BGR for OpenCV compat
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception as e:
            log.error("Picamera2 capture error: %s", e)
            return False, None

    def release(self):
        try:
            self._picam.stop()
            self._picam.close()
        except Exception:
            pass

    def isOpened(self):
        return True


# ─────────────────────────────────────────────────────────────────────
# Frame reading helper
# ─────────────────────────────────────────────────────────────────────

def read_frame(camera) -> tuple:
    """
    Read a single frame, applying 180° rotation if configured.

    Returns:
        (success: bool, frame: np.ndarray or None)
    """
    ret, frame = camera.read()
    if ret and frame is not None and CAMERA_ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return ret, frame


def flush_camera_buffer(camera, count: int = 3):
    """Discard stale buffered frames after servo movement."""
    log.debug("Flushing %d stale camera frames", count)
    for _ in range(count):
        read_frame(camera)
