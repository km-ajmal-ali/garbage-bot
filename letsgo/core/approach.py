"""
WasteBot Approach Module
=========================
Handles the APPROACH state – driving the robot toward a detected
object while keeping it horizontally centred in the frame.
"""

import time

from core.config import (
    APPROACH_SPEED,
    CENTER_TOLERANCE_X,
    DEPTH_APPROACH_STOP,
    MAX_LOST_FRAMES,
    STEER_PULSE_TIME,
    DRIVE_PULSE_TIME,
    CAM_WIDTH,
)
from core.states import STATE_SCANNING, STATE_TILT_ADJUST
from core.detection import run_detection, pick_best_target, estimate_object_depth


class Approacher:
    """Drives the chassis toward the best-detected object."""

    def __init__(self, model, camera, motors, display):
        self.model   = model
        self.camera  = camera
        self.motors  = motors
        self.display = display

    def execute(self, running_flag: callable) -> tuple[str, dict | None]:
        """
        Run the approach loop until the robot is close enough,
        the target is lost, or the running flag becomes False.

        Args:
            running_flag: Callable returning bool (True → keep going).

        Returns:
            (next_state, last_target_detection_or_None)
        """
        lost_count = 0
        target = None

        while running_flag():
            ret, frame = self.camera.read()
            if not ret or frame is None:
                continue

            detections = run_detection(self.model, frame)
            self.display.show(frame, detections, "APPROACH")

            target = pick_best_target(detections)

            if target is None:
                lost_count += 1
                if lost_count >= MAX_LOST_FRAMES:
                    print("[APPROACH] Target lost → back to SCANNING.")
                    self.motors.stop()
                    return STATE_SCANNING, None
                continue

            lost_count = 0

            # ── Depth check (do this first so we don't steer unnecessarily) ─
            depth = estimate_object_depth(target['width_px'])
            target['depth_cm'] = depth

            if 0 < depth <= DEPTH_APPROACH_STOP:
                print("[APPROACH] Close enough → TILT_ADJUST.")
                self.motors.stop()
                return STATE_TILT_ADJUST, target

            # ── Horizontal alignment ──────────────────────────────────
            cx_frac  = target['center'][0] / CAM_WIDTH
            offset_x = cx_frac - 0.5

            if offset_x < -CENTER_TOLERANCE_X:
                self.motors.move("left", APPROACH_SPEED)
                time.sleep(STEER_PULSE_TIME)
                self.motors.stop()
            elif offset_x > CENTER_TOLERANCE_X:
                self.motors.move("right", APPROACH_SPEED)
                time.sleep(STEER_PULSE_TIME)
                self.motors.stop()
            else:
                # Aligned – drive forward
                self.motors.move("forward", APPROACH_SPEED)
                time.sleep(DRIVE_PULSE_TIME)

            # No extra sleep – the next camera.read() + inference
            # naturally paces the loop at ~15 FPS

        self.motors.stop()
        return STATE_SCANNING, target
