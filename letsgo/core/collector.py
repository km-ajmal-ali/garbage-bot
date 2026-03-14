"""
WasteBot Collector Module
==========================
Handles the COLLECT state – slowly driving the chassis forward
over the detected object so the passive collector picks it up.
"""

import time

from core.config import (
    COLLECT_SPEED, DEPTH_COLLECT_DONE, MAX_LOST_COLLECT,
    PAN_CENTER_ANGLE, TILT_CENTER_ANGLE,
)
from core.states import STATE_SCANNING
from core.detection import run_detection, pick_best_target, estimate_object_depth
from core.logger import get_logger

log = get_logger("collect")


class Collector:
    """Drives the robot over the target object and resets for next scan."""

    def __init__(self, model, camera, motors, pan_servo, tilt_servo, display):
        self.model      = model
        self.camera     = camera
        self.motors     = motors
        self.pan_servo  = pan_servo
        self.tilt_servo = tilt_servo
        self.display    = display
        log.info("Collector initialised (speed=%d%%, depth_done=%dcm, max_lost=%d)",
                 COLLECT_SPEED, DEPTH_COLLECT_DONE, MAX_LOST_COLLECT)

    def execute(self, running_flag: callable) -> str:
        """
        Crawl forward until the object disappears from view
        (collected) or is extremely close.

        Returns:
            Next state (always STATE_SCANNING).
        """
        log.info("═══ COLLECT started ═══")
        log.info("Driving FORWARD at %d%% speed", COLLECT_SPEED)
        self.motors.move("forward", COLLECT_SPEED)

        lost_count = 0
        frame_num = 0

        while running_flag():
            ret, frame = self.camera.read()
            if not ret or frame is None:
                log.warning("Camera read failed during collection")
                continue

            frame_num += 1
            detections = run_detection(self.model, frame, camera=self.camera)
            self.display.show(frame, detections, "COLLECT")

            target = pick_best_target(detections)

            if target is None:
                lost_count += 1
                log.debug("Frame %d: target lost (lost_count=%d/%d)",
                          frame_num, lost_count, MAX_LOST_COLLECT)
                if lost_count >= MAX_LOST_COLLECT:
                    log.info("✓ Object not visible for %d frames – COLLECTED!", lost_count)
                    break
                continue

            lost_count = 0
            depth = estimate_object_depth(target['width_px'])

            log.info("Frame %d: '%s' still visible  depth≈%.0fcm  (done at ≤%dcm)",
                     frame_num, target['label'], depth, DEPTH_COLLECT_DONE)

            if 0 < depth <= DEPTH_COLLECT_DONE:
                log.info("✓ Object very close (%.0fcm ≤ %dcm) – COLLECTED!", depth, DEPTH_COLLECT_DONE)
                break

        # ── Reset hardware ────────────────────────────────────────────
        log.info("Stopping motors and resetting servos to centre")
        self.motors.stop()
        self.tilt_servo.move_and_detach(TILT_CENTER_ANGLE, settle=0.15)
        self.pan_servo.move_and_detach(PAN_CENTER_ANGLE, settle=0.15)

        log.info("═══ COLLECT done (processed %d frames) → SCANNING ═══\n", frame_num)
        return STATE_SCANNING
