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
from core.logger import get_logger

log = get_logger("navigate")

# Flush stale frames after chassis rotation
BUFFER_FLUSH_FRAMES = 3


class Navigator:
    """Rotates the chassis to search for objects in new directions."""

    def __init__(self, model, camera, motors, display):
        self.model   = model
        self.camera  = camera
        self.motors  = motors
        self.display = display
        self.search_steps_done = 0
        log.info("Navigator initialised (speed=%d%%, step_time=%.1fs, max_steps=%d)",
                 SEARCH_ROTATE_SPEED, ROTATE_STEP_TIME, MAX_SEARCH_STEPS)

    def reset(self):
        """Reset the rotation counter (e.g. after a successful find)."""
        log.info("Search rotation counter reset (was %d)", self.search_steps_done)
        self.search_steps_done = 0

    def step(self) -> tuple[str, dict | None]:
        """
        Rotate the chassis one step, then quick-check for objects.

        Returns:
            (next_state, target_detection_or_None)
        """
        if self.search_steps_done >= MAX_SEARCH_STEPS:
            log.warning("Full rotation complete (%d steps) – restarting search",
                        MAX_SEARCH_STEPS)
            self.search_steps_done = 0

        # Alternate direction after each full circle
        half = MAX_SEARCH_STEPS
        direction = ("right"
                     if (self.search_steps_done % (half * 2)) < half
                     else "left")

        log.info("SEARCH step %d/%d → rotating %s at %d%% for %.1fs",
                 self.search_steps_done + 1, MAX_SEARCH_STEPS,
                 direction, SEARCH_ROTATE_SPEED, ROTATE_STEP_TIME)

        self.motors.move(direction, SEARCH_ROTATE_SPEED)
        time.sleep(ROTATE_STEP_TIME)
        self.motors.stop()
        time.sleep(0.2)
        log.debug("Rotation complete, motors stopped")

        self.search_steps_done += 1

        # Flush stale camera frames after chassis rotation
        log.debug("Flushing %d stale camera frames after rotation", BUFFER_FLUSH_FRAMES)
        for _ in range(BUFFER_FLUSH_FRAMES):
            self.camera.read()

        # Quick detection check after rotation
        ret, frame = self.camera.read()
        if ret and frame is not None:
            detections = run_detection(self.model, frame)
            self.display.show(frame, detections, "SEARCH_ROTATE")

            log.info("Post-rotation check: %d detection(s)", len(detections))

            if detections:
                target = pick_best_target(detections)
                if target:
                    log.info("✓ FOUND '%s' conf=%.2f during search rotation → APPROACH",
                             target['label'], target['confidence'])
                    return STATE_APPROACH, target
        else:
            log.warning("Camera read failed after rotation")

        log.debug("Nothing found → back to SCANNING sweep")
        return STATE_SCANNING, None
