"""
WasteBot Collector Module
==========================
Handles the COLLECT state – slowly driving the chassis forward
over the detected object so the passive collector picks it up.
"""

import time

from core.config import COLLECT_SPEED, DEPTH_COLLECT_DONE, MAX_LOST_COLLECT
from core.states import STATE_SCANNING
from core.detection import run_detection, pick_best_target, estimate_object_depth


class Collector:
    """Drives the robot over the target object and resets for next scan."""

    def __init__(self, model, camera, motors, pan_servo, tilt_servo, display):
        self.model      = model
        self.camera     = camera
        self.motors     = motors
        self.pan_servo  = pan_servo
        self.tilt_servo = tilt_servo
        self.display    = display

    def execute(self, running_flag: callable) -> str:
        """
        Crawl forward until the object disappears from view
        (collected) or is extremely close.

        Returns:
            Next state (always STATE_SCANNING).
        """
        print("[COLLECT] Driving over object to collect …")
        self.motors.move("forward", COLLECT_SPEED)

        lost_count = 0

        while running_flag():
            ret, frame = self.camera.read()
            if not ret or frame is None:
                continue

            detections = run_detection(self.model, frame)
            self.display.show(frame, detections, "COLLECT")

            target = pick_best_target(detections)

            if target is None:
                lost_count += 1
                if lost_count >= MAX_LOST_COLLECT:
                    print("[COLLECT] Object no longer visible – collected!")
                    break
                continue

            lost_count = 0
            depth = estimate_object_depth(target['width_px'])

            if 0 < depth <= DEPTH_COLLECT_DONE:
                print(f"[COLLECT] Object very close ({depth:.0f} cm) – collected!")
                break

            # No sleep – camera.read() + inference naturally paces the loop

        # ── Reset hardware ────────────────────────────────────────────
        self.motors.stop()
        self.tilt_servo.servo.angle = 0
        self.pan_servo.servo.angle = 0
        time.sleep(0.15)

        print("[COLLECT] Done.  Returning to SCANNING.\n")
        return STATE_SCANNING
