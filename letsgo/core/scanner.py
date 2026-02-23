"""
WasteBot Scanner Module
========================
Handles the SCANNING state – panning the camera through preset
positions and checking for objects at each stop.
"""

import time

from core.config import SCAN_POSITIONS, SCAN_DWELL
from core.states import STATE_APPROACH, STATE_SEARCH_ROTATE
from core.detection import run_detection, pick_best_target


class Scanner:
    """Controls the camera-pan scanning sweep."""

    def __init__(self, model, camera, pan_servo, display):
        self.model     = model
        self.camera    = camera
        self.pan_servo = pan_servo
        self.display   = display
        self.scan_index = 0
        self._last_angle = None

    def reset(self):
        """Restart the sweep from the first position."""
        self.scan_index = 0

    def step(self, current_state: str) -> tuple[str, dict | None]:
        """
        Execute one step of the scanning sweep.

        Returns:
            (next_state, target_detection_or_None)
        """
        if self.scan_index >= len(SCAN_POSITIONS):
            print("[SCAN] Full sweep complete – no objects detected.")
            self.reset()
            return STATE_SEARCH_ROTATE, None

        angle = SCAN_POSITIONS[self.scan_index]

        # Only sleep if the servo actually needs to move
        if angle != self._last_angle:
            self.pan_servo.servo.angle = angle   # direct set, skip the 0.3s sleep in set_angle
            self._last_angle = angle
            time.sleep(SCAN_DWELL)               # short settle (0.15s)

        ret, frame = self.camera.read()
        if not ret or frame is None:
            self.scan_index += 1
            return current_state, None

        detections = run_detection(self.model, frame)
        self.display.show(frame, detections, current_state)

        if detections:
            target = pick_best_target(detections)
            if target:
                depth = target.get('depth_cm', -1)
                print(f"[SCAN] Object '{target['label']}' detected at "
                      f"pan={angle}°  depth≈{depth:.0f} cm")
                return STATE_APPROACH, target

        self.scan_index += 1
        return current_state, None
