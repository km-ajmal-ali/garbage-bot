"""
WasteBot Navigator Module
===========================
Handles the SEARCH_ROTATE state – rotating the chassis in place
when the camera sweep found nothing, then re-scanning.
"""

import time

from core.config import SEARCH_ROTATE_SPEED, ROTATE_STEP_TIME, MAX_SEARCH_STEPS
from core.states import STATE_SCANNING, STATE_APPROACH
from core.detection import run_detection, pick_best_target


class Navigator:
    """Rotates the chassis to search for objects in new directions."""

    def __init__(self, model, camera, motors, display):
        self.model   = model
        self.camera  = camera
        self.motors  = motors
        self.display = display
        self.search_steps_done = 0

    def reset(self):
        """Reset the rotation counter (e.g. after a successful find)."""
        self.search_steps_done = 0

    def step(self) -> tuple[str, dict | None]:
        """
        Rotate the chassis one step, then quick-check for objects.

        Returns:
            (next_state, target_detection_or_None)
        """
        if self.search_steps_done >= MAX_SEARCH_STEPS:
            print("[SEARCH] Full rotation complete – trying again.")
            self.search_steps_done = 0

        # Alternate direction after each full circle
        half = MAX_SEARCH_STEPS
        direction = ("right"
                     if (self.search_steps_done % (half * 2)) < half
                     else "left")

        print(f"[SEARCH] Rotating {direction} "
              f"(step {self.search_steps_done + 1}/{MAX_SEARCH_STEPS}) …")

        self.motors.move(direction, SEARCH_ROTATE_SPEED)
        time.sleep(ROTATE_STEP_TIME)
        self.motors.stop()
        time.sleep(0.2)

        self.search_steps_done += 1

        # Quick detection check after rotation
        ret, frame = self.camera.read()
        if ret and frame is not None:
            detections = run_detection(self.model, frame)
            self.display.show(frame, detections, "SEARCH_ROTATE")

            if detections:
                target = pick_best_target(detections)
                if target:
                    print(f"[SEARCH] Found '{target['label']}' during rotation!")
                    return STATE_APPROACH, target

        # Nothing found – go back to a full pan scan
        return STATE_SCANNING, None
