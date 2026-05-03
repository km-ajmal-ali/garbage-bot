"""
WasteBot QR Navigator Module
=============================
Handles the SEARCH_ROTATE_QR state – rotating the chassis to find a QR code.
"""

import time

from core.config import SEARCH_ROTATE_SPEED, ROTATE_STEP_TIME, MAX_SEARCH_STEPS
from core.states import STATE_SEARCH_QR, STATE_APPROACH_QR
from core.detection import detect_qr_code, read_frame
from core.logger import get_logger

log = get_logger("qr_navigator")

BUFFER_FLUSH_FRAMES = 3

class QRNavigator:
    def __init__(self, camera, motors, display):
        self.camera  = camera
        self.motors  = motors
        self.display = display
        self.search_steps_done = 0
        log.info("QR Navigator initialised")

    def reset(self):
        self.search_steps_done = 0

    def step(self) -> tuple[str, dict | None]:
        if self.search_steps_done >= MAX_SEARCH_STEPS:
            log.warning("Full QR rotation complete – restarting QR search")
            self.search_steps_done = 0

        half = MAX_SEARCH_STEPS
        direction = "right" if (self.search_steps_done % (half * 2)) < half else "left"

        log.info("QR SEARCH step %d/%d → rotating %s",
                 self.search_steps_done + 1, MAX_SEARCH_STEPS, direction)

        self.motors.move(direction, SEARCH_ROTATE_SPEED)
        time.sleep(ROTATE_STEP_TIME)
        self.motors.stop()
        time.sleep(0.2)

        self.search_steps_done += 1

        for _ in range(BUFFER_FLUSH_FRAMES):
            read_frame(self.camera)

        ret, frame = read_frame(self.camera)
        if ret and frame is not None:
            target = detect_qr_code(frame)
            self.display.show(frame, [target] if target else [], "SEARCH_ROTATE_QR")

            if target:
                log.info("✓ QR CODE FOUND during rotation → APPROACH_QR")
                return STATE_APPROACH_QR, target

        return STATE_SEARCH_QR, None
