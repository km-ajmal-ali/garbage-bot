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
from core.logger import get_logger

log = get_logger("approach")


class Approacher:
    """Drives the chassis toward the best-detected object."""

    def __init__(self, model, camera, motors, display):
        self.model   = model
        self.camera  = camera
        self.motors  = motors
        self.display = display
        log.info("Approacher initialised (speed=%d%%, depth_stop=%dcm, tolerance_x=%.2f)",
                 APPROACH_SPEED, DEPTH_APPROACH_STOP, CENTER_TOLERANCE_X)

    def execute(self, running_flag: callable) -> tuple[str, dict | None]:
        """
        Run the approach loop until the robot is close enough,
        the target is lost, or the running flag becomes False.

        Returns:
            (next_state, last_target_detection_or_None)
        """
        log.info("═══ APPROACH started ═══")
        lost_count = 0
        target = None
        frame_num = 0

        while running_flag():
            ret, frame = self.camera.read()
            if not ret or frame is None:
                log.warning("Camera read failed in approach loop")
                continue

            frame_num += 1
            detections = run_detection(self.model, frame, camera=self.camera)
            self.display.show(frame, detections, "APPROACH")

            target = pick_best_target(detections)

            if target is None:
                lost_count += 1
                log.debug("Frame %d: no target (lost_count=%d/%d)",
                          frame_num, lost_count, MAX_LOST_FRAMES)
                if lost_count >= MAX_LOST_FRAMES:
                    log.warning("Target lost for %d consecutive frames → SCANNING", lost_count)
                    self.motors.stop()
                    return STATE_SCANNING, None
                continue

            lost_count = 0

            # ── Depth check (do this first so we don't steer unnecessarily) ─
            depth = estimate_object_depth(target['width_px'])
            target['depth_cm'] = depth

            # ── Horizontal alignment ──────────────────────────────────
            cx_frac  = target['center'][0] / CAM_WIDTH
            offset_x = cx_frac - 0.5

            log.info("Frame %d: '%s' conf=%.2f  depth≈%.0fcm  offset_x=%+.2f  bbox_w=%dpx",
                     frame_num, target['label'], target['confidence'],
                     depth, offset_x, target['width_px'])

            if 0 < depth <= DEPTH_APPROACH_STOP:
                log.info("✓ CLOSE ENOUGH (%.0fcm ≤ %dcm) → TILT_ADJUST", depth, DEPTH_APPROACH_STOP)
                self.motors.stop()
                return STATE_TILT_ADJUST, target

            if offset_x < -CENTER_TOLERANCE_X:
                log.debug("Steering LEFT (offset=%.2f < -%.2f)", offset_x, CENTER_TOLERANCE_X)
                self.motors.move("left", APPROACH_SPEED)
                time.sleep(STEER_PULSE_TIME)
                self.motors.stop()
            elif offset_x > CENTER_TOLERANCE_X:
                log.debug("Steering RIGHT (offset=%.2f > +%.2f)", offset_x, CENTER_TOLERANCE_X)
                self.motors.move("right", APPROACH_SPEED)
                time.sleep(STEER_PULSE_TIME)
                self.motors.stop()
            else:
                log.debug("Aligned – driving FORWARD (offset=%.2f within ±%.2f)",
                          offset_x, CENTER_TOLERANCE_X)
                self.motors.move("forward", APPROACH_SPEED)
                time.sleep(DRIVE_PULSE_TIME)

        log.warning("Approach loop exited (running=False)")
        self.motors.stop()
        return STATE_SCANNING, target
