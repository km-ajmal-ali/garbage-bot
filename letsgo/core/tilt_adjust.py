"""
WasteBot Tilt-Adjust Module
=============================
Handles the TILT_ADJUST state – tilting the camera downward so
the nearby object remains visible in the frame before collection.
"""

import time

from core.config import MAX_TILT_DOWN, TILT_STEP, CAM_HEIGHT
from core.states import STATE_COLLECT
from core.detection import run_detection, pick_best_target


class TiltAdjuster:
    """Tilts the Y-axis servo to keep the object centred vertically."""

    def __init__(self, model, camera, tilt_servo, display):
        self.model      = model
        self.camera     = camera
        self.tilt_servo = tilt_servo
        self.display    = display

    def execute(self, running_flag: callable) -> tuple[str, dict | None]:
        """
        Incrementally tilt the camera downward until the object
        is vertically centred or the maximum tilt is reached.

        Returns:
            (next_state, last_target_detection_or_None)
        """
        print("[TILT] Adjusting camera tilt to view nearby object …")

        tilt_angle = 0
        target = None

        while tilt_angle >= MAX_TILT_DOWN and running_flag():
            self.tilt_servo.set_angle(tilt_angle)
            time.sleep(0.3)

            ret, frame = self.camera.read()
            if not ret or frame is None:
                tilt_angle += TILT_STEP
                continue

            detections = run_detection(self.model, frame)
            self.display.show(frame, detections, "TILT_ADJUST")

            target = pick_best_target(detections)
            if target:
                cy_frac = target['center'][1] / CAM_HEIGHT
                print(f"[TILT] tilt={tilt_angle}°  cy_frac={cy_frac:.2f}")

                if 0.3 <= cy_frac <= 0.7:
                    print("[TILT] Object centred in frame → COLLECT.")
                    return STATE_COLLECT, target

            tilt_angle += TILT_STEP

        print("[TILT] Max tilt reached → COLLECT.")
        return STATE_COLLECT, target
