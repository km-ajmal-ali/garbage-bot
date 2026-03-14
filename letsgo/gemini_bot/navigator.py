"""
GeminiBot Navigator Module
============================
Handles the SEARCH_ROTATE state — rotates the chassis in place
when the camera sweep found nothing.  After each rotation step,
does a quick Gemini check to see if anything new appeared.
"""

import time

from core.logger import get_logger
from gemini_bot.config import (
    SEARCH_ROTATE_SPEED,
    ROTATE_STEP_TIME,
    MAX_SEARCH_STEPS,
)
from gemini_bot.camera import read_frame, flush_camera_buffer
from gemini_bot.gemini_vision import scan_image

log = get_logger("g_navigate")


class GeminiNavigator:
    """Rotates the chassis to search for objects in new directions."""

    def __init__(self, camera, motors):
        self.camera = camera
        self.motors = motors
        self.search_steps_done = 0
        log.info("GeminiNavigator initialised (speed=%d%%, step_time=%.1fs, max=%d)",
                 SEARCH_ROTATE_SPEED, ROTATE_STEP_TIME, MAX_SEARCH_STEPS)

    def reset(self):
        """Reset the rotation counter."""
        log.info("Search rotation counter reset (was %d)", self.search_steps_done)
        self.search_steps_done = 0

    def step(self) -> tuple:
        """
        Rotate the chassis one step, then quick-check with Gemini.

        Returns:
            (next_state, target_info_or_None)
        """
        if self.search_steps_done >= MAX_SEARCH_STEPS:
            log.warning("Full rotation done (%d steps) → restart", MAX_SEARCH_STEPS)
            self.search_steps_done = 0

        # Alternate direction each half circle
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
        time.sleep(0.3)

        self.search_steps_done += 1

        # Flush old frames after rotation
        flush_camera_buffer(self.camera, 3)

        # Quick Gemini check
        ret, frame = read_frame(self.camera)
        if ret and frame is not None:
            result = scan_image(frame)
            if result.get("found", False) and result.get("objects"):
                best = result["objects"][0]
                log.info("✓ FOUND '%s' during search rotation → APPROACH",
                         best.get("label", "?"))
                return "APPROACH", {
                    "angle": 0,   # chassis is already facing it
                    "objects": result["objects"],
                    "best": best,
                }
        else:
            log.warning("Camera read failed after rotation")

        log.debug("Nothing found → back to SCANNING sweep")
        return "SCANNING", None
