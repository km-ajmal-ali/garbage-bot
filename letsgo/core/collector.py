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
    MAX_OPEN_ANGLE, MAX_CLOSE_ANGLE,
)
from core.states import STATE_SCANNING, STATE_SEARCH_ROTATE
from core.detection import run_detection, pick_best_target, estimate_object_depth, read_frame
from core.logger import get_logger

log = get_logger("collect")


class Collector:
    """Drives the robot over the target object and resets for next scan."""

    def __init__(self, model, camera, motors, pan_servo, tilt_servo, collector_servo, display):
        self.model           = model
        self.camera          = camera
        self.motors          = motors
        self.pan_servo       = pan_servo
        self.tilt_servo      = tilt_servo
        self.collector_servo = collector_servo
        self.display         = display
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
        log.info("Ensuring gripper is open")
        self.collector_servo.move_and_detach(MAX_OPEN_ANGLE, settle=0.3)
        log.info("Driving FORWARD at %d%% speed", COLLECT_SPEED)
        self.motors.move("forward", COLLECT_SPEED)

        lost_count = 0
        frame_num = 0

        while running_flag():
            ret, frame = read_frame(self.camera)
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

        # ── Gripper sequence ──────────────────────────────────────────
        log.info("Stopping motors, target collected or lost")
        self.motors.stop()
        time.sleep(0.5)

        log.info("Closing gripper around object")
        self.collector_servo.move_and_detach(MAX_CLOSE_ANGLE, settle=1.0)
        
        log.info("Driving forward softly with object")
        self.motors.move("forward", COLLECT_SPEED)
        time.sleep(1.5)
        self.motors.stop()
        time.sleep(0.5)

        log.info("Opening gripper to release")
        self.collector_servo.move_and_detach(MAX_OPEN_ANGLE, settle=1.0)

        log.info("Moving backward slightly before search rotation")
        self.motors.move("backward", COLLECT_SPEED)
        time.sleep(1.0)
        self.motors.stop()

        # ── Reset hardware ────────────────────────────────────────────
        log.info("Resetting servos to centre")
        self.tilt_servo.move_and_detach(TILT_CENTER_ANGLE, settle=0.15)
        self.pan_servo.move_and_detach(PAN_CENTER_ANGLE, settle=0.15)

        log.info("═══ COLLECT done (processed %d frames) → SEARCH_ROTATE ═══\n", frame_num)
        return STATE_SEARCH_ROTATE
