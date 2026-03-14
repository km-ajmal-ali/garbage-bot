"""
WasteBot Tilt-Adjust Module
=============================
Handles the TILT_ADJUST state – tilting the camera downward so
the nearby object remains visible in the frame before collection.
"""

import time

from core.config import TILT_MIN_ANGLE, TILT_CENTER_ANGLE, TILT_STEP, TILT_SETTLE, CAM_HEIGHT
from core.states import STATE_COLLECT
from core.detection import run_detection, pick_best_target
from core.logger import get_logger

log = get_logger("tilt")


class TiltAdjuster:
    """Tilts the Y-axis servo to keep the object centred vertically."""

    def __init__(self, model, camera, tilt_servo, display):
        self.model      = model
        self.camera     = camera
        self.tilt_servo = tilt_servo
        self.display    = display
        log.info("TiltAdjuster initialised (min=%d°, center=%d°, step=%d°, settle=%.2fs)",
                 TILT_MIN_ANGLE, TILT_CENTER_ANGLE, TILT_STEP, TILT_SETTLE)

    def execute(self, running_flag: callable) -> tuple[str, dict | None]:
        """
        Incrementally tilt the camera downward until the object
        is vertically centred or the maximum tilt is reached.

        Returns:
            (next_state, last_target_detection_or_None)
        """
        log.info("═══ TILT_ADJUST started ═══")

        tilt_angle = TILT_CENTER_ANGLE
        target = None
        step_num = 0

        while tilt_angle >= TILT_MIN_ANGLE and running_flag():
            step_num += 1

            # Clamp to configured tilt limits to prevent over-rotation
            tilt_angle = self.tilt_servo.clamp_angle(tilt_angle)

            log.info("Tilt step %d: servo → %d°", step_num, tilt_angle)
            self.tilt_servo.servo.angle = tilt_angle
            time.sleep(TILT_SETTLE)

            ret, frame = self.camera.read()
            if not ret or frame is None:
                log.warning("Camera read failed at tilt=%d°", tilt_angle)
                tilt_angle += TILT_STEP
                continue

            detections = run_detection(self.model, frame, camera=self.camera)
            self.display.show(frame, detections, "TILT_ADJUST")

            target = pick_best_target(detections)
            if target:
                cy_frac = target['center'][1] / CAM_HEIGHT
                log.info("  target '%s' at tilt=%d°  cy_frac=%.2f  (want 0.30–0.70)",
                         target['label'], tilt_angle, cy_frac)

                if 0.3 <= cy_frac <= 0.7:
                    log.info("✓ Object vertically centred (cy=%.2f) → COLLECT", cy_frac)
                    return STATE_COLLECT, target
                else:
                    log.debug("  object not centred (cy=%.2f) → tilting further", cy_frac)
            else:
                log.debug("  no target found at tilt=%d°", tilt_angle)

            tilt_angle += TILT_STEP

        log.warning("Max tilt (%d°) reached without centring → COLLECT anyway", TILT_MIN_ANGLE)
        return STATE_COLLECT, target
