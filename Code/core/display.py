"""
WasteBot Display Module
========================
Handles the optional OpenCV debug window – drawing bounding boxes,
HUD overlay, and the current state label.

Safe to use in headless mode; display errors are silently ignored.
"""

import time
import numpy as np

from test.bounding_box import draw_bounding_boxes, draw_hud
from core.config import CAM_HEIGHT, FRAME_SKIP


class Display:
    """Manages the debug display window and FPS tracking."""

    def __init__(self):
        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.time()

    def show(self, frame: np.ndarray, detections: list, state: str):
        """
        Draw overlays and show the frame in an OpenCV window.
        Only processes every FRAME_SKIP-th frame to reduce overhead.

        Args:
            frame:      BGR image (numpy array).
            detections: List of detection dicts.
            state:      Current FSM state string (shown on screen).
        """
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP != 0:
            return

        now = time.time()
        self.fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

        try:
            import cv2

            draw_bounding_boxes(frame, detections, show_depth=True)
            draw_hud(frame, self.fps, len(detections))

            cv2.putText(
                frame, f"State: {state}", (10, CAM_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                cv2.LINE_AA,
            )

            cv2.imshow("WasteBot View", frame)
            cv2.waitKey(1)
        except Exception:
            pass  # headless / no display – silently skip
