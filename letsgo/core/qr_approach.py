"""
WasteBot QR Approach Module
============================
Handles the APPROACH_QR state – driving towards a detected QR code.
"""

import time

from core.config import (
    APPROACH_SPEED,
    CENTER_TOLERANCE_X,
    DEPTH_QR_APPROACH_STOP,
    MAX_LOST_FRAMES,
    STEER_PULSE_TIME,
    DRIVE_PULSE_TIME,
    CAM_WIDTH,
    CHASSIS_DEGREES_PER_SEC,
)
from core.states import STATE_SEARCH_QR, STATE_DROP_QR
from core.detection import detect_qr_code, read_frame
from core.logger import get_logger

log = get_logger("qr_approach")

class QRApproacher:
    def __init__(self, camera, motors, display):
        self.camera  = camera
        self.motors  = motors
        self.display = display
        log.info("QR Approacher initialised")

    def execute(self, running_flag: callable) -> tuple[str, dict | None]:
        log.info("═══ APPROACH_QR started ═══")
        lost_count = 0
        target = None
        frame_num = 0
        self.last_depth = 999.0
        self.last_target = None

        while running_flag():
            ret, frame = read_frame(self.camera)
            if not ret or frame is None:
                continue

            frame_num += 1
            target = detect_qr_code(frame)
            self.display.show(frame, [target] if target else [], "APPROACH_QR")

            if target is None:
                self.motors.stop()
                lost_count += 1
                if lost_count >= MAX_LOST_FRAMES:
                    if self.last_depth <= DEPTH_QR_APPROACH_STOP + 25:
                        log.warning("QR lost but close → DROP_QR")
                        return STATE_DROP_QR, self.last_target
                    log.warning("QR lost → SEARCH_QR")
                    return STATE_SEARCH_QR, None
                continue

            lost_count = 0
            self.last_target = target
            depth = target['depth_cm']
            self.last_depth = depth

            cx_frac = target['center'][0] / CAM_WIDTH
            offset_x = cx_frac - 0.5

            if 0 < depth <= DEPTH_QR_APPROACH_STOP:
                log.info("✓ REACHED QR DESTINATION (%.0fcm) → DROP_QR", depth)
                self.motors.stop()
                return STATE_DROP_QR, target

            if offset_x < -CENTER_TOLERANCE_X or offset_x > CENTER_TOLERANCE_X:
                # Proportional steering: calculate turn time just like we do for pan angle!
                angle = abs(offset_x) * 62.0  # Approximate horizontal FOV is 62 degrees
                turn_time = angle / CHASSIS_DEGREES_PER_SEC
                direction = "left" if offset_x > 0 else "right"
                
                log.debug("Steering %s proportionally for %.2fs (offset=%.2f, angle=%.1f°)", 
                          direction, turn_time, offset_x, angle)
                self.motors.move(direction, APPROACH_SPEED)
                time.sleep(turn_time)
                self.motors.stop()
            else:
                self.motors.move("forward", APPROACH_SPEED)
                time.sleep(DRIVE_PULSE_TIME)

        self.motors.stop()
        return STATE_SEARCH_QR, target
