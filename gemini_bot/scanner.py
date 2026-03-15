"""
GeminiBot Scanner Module
=========================
Handles the SCANNING state – pans the camera servo through preset
positions, captures a frame at each stop, sends it to Gemini to
check for garbage / waste objects.

Cost optimisation:
  • Only 1 image per scan position (configurable)
  • Images are resized + JPEG-compressed before sending
  • Short, structured prompts that return minimal JSON
"""

import time

from core.logger import get_logger
from gemini_bot.config import (
    SCAN_POSITIONS, SCAN_DWELL,
    SCAN_SAMPLES_PER_POSITION,
    SCAN_IMAGE_INTERVAL,
)
from gemini_bot.camera import read_frame, flush_camera_buffer
from gemini_bot.gemini_vision import scan_image

log = get_logger("g_scanner")

# Stale frames to discard after servo movement
BUFFER_FLUSH_FRAMES = 3


class GeminiScanner:
    """Pans the servo and uses Gemini to detect garbage at each position."""

    def __init__(self, camera, pan_servo):
        self.camera    = camera
        self.pan_servo = pan_servo
        self.scan_index = 0
        log.info("GeminiScanner initialised with %d positions: %s",
                 len(SCAN_POSITIONS), SCAN_POSITIONS)

    def reset(self):
        """Restart the sweep from the first position."""
        log.info("Scan sweep reset → starting from position 0")
        self.scan_index = 0

    def step(self) -> tuple:
        """
        Execute one step of the scanning sweep.

        Returns:
            (next_state, result_dict_or_None)
              next_state: "SCANNING", "APPROACH", or "SEARCH_ROTATE"
              result_dict: {"angle": int, "objects": [...]} if found
        """
        if self.scan_index >= len(SCAN_POSITIONS):
            log.warning("Full sweep complete – no objects found across %d positions",
                        len(SCAN_POSITIONS))
            self.reset()
            return "SEARCH_ROTATE", None

        angle = SCAN_POSITIONS[self.scan_index]
        angle = self.pan_servo.clamp_angle(angle)

        # ── Move servo, settle, detach PWM ─────────────────────────────
        log.info("SCAN step %d/%d → pan servo to %d°",
                 self.scan_index + 1, len(SCAN_POSITIONS), angle)
        self.pan_servo.move_and_detach(angle, settle=SCAN_DWELL)

        # ── Flush stale buffered frames ────────────────────────────────
        flush_camera_buffer(self.camera, BUFFER_FLUSH_FRAMES)

        # ── Capture and analyse ────────────────────────────────────────
        for sample in range(SCAN_SAMPLES_PER_POSITION):
            ret, frame = read_frame(self.camera)
            if not ret or frame is None:
                log.warning("Camera read failed at pan=%d° sample=%d",
                            angle, sample + 1)
                continue

            log.info("Sending frame to Gemini (pan=%d° sample=%d/%d)",
                     angle, sample + 1, SCAN_SAMPLES_PER_POSITION)

            result = scan_image(frame)

            if result.get("found", False) and result.get("objects"):
                best = result["objects"][0]
                log.info("✓ GARBAGE FOUND at pan=%d°: '%s' (conf=%.0f%%, pos=%s, dist=%s) → APPROACH",
                         angle, best.get("label", "?"),
                         best.get("confidence", 0) * 100,
                         best.get("position", "?"),
                         best.get("distance", "?"))
                return "APPROACH", {
                    "angle": angle,
                    "objects": result["objects"],
                    "best": best,
                }
            else:
                log.info("No garbage detected at pan=%d° (sample %d)",
                         angle, sample + 1)

            # Wait between samples at the same position
            if sample < SCAN_SAMPLES_PER_POSITION - 1:
                time.sleep(SCAN_IMAGE_INTERVAL)

        # ── No detection at this position → advance ────────────────────
        self.scan_index += 1
        return "SCANNING", None
