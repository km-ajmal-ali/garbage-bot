"""
WasteBot QR Scanner Module
==========================
Handles the SEARCH_QR state – panning the camera to find a QR code.
"""

import time

from core.config import SCAN_POSITIONS, SCAN_DWELL
from core.states import STATE_APPROACH_QR, STATE_SEARCH_ROTATE_QR
from core.detection import detect_qr_code, read_frame
from core.logger import get_logger

log = get_logger("qr_scanner")

BUFFER_FLUSH_FRAMES = 3
SAMPLES_PER_POSITION = 5

class QRScanner:
    def __init__(self, camera, pan_servo, display):
        self.camera    = camera
        self.pan_servo = pan_servo
        self.display   = display
        self.scan_index = 0
        log.info("QR Scanner initialised")

    def reset(self):
        self.scan_index = 0

    def _flush_camera_buffer(self):
        for _ in range(BUFFER_FLUSH_FRAMES):
            read_frame(self.camera)

    def step(self, current_state: str) -> tuple[str, dict | None]:
        if self.scan_index >= len(SCAN_POSITIONS):
            log.warning("Full QR sweep complete – no QR detected")
            self.reset()
            return STATE_SEARCH_ROTATE_QR, None

        angle = SCAN_POSITIONS[self.scan_index]
        angle = self.pan_servo.clamp_angle(angle)

        log.info("QR SCAN step %d/%d → pan servo to %d°",
                 self.scan_index + 1, len(SCAN_POSITIONS), angle)
        self.pan_servo.move_and_detach(angle, settle=SCAN_DWELL)

        self._flush_camera_buffer()

        for attempt in range(SAMPLES_PER_POSITION):
            ret, frame = read_frame(self.camera)
            if not ret or frame is None:
                continue

            target = detect_qr_code(frame)
            
            # Show empty or QR detection
            self.display.show(frame, [target] if target else [], current_state)

            if target:
                target['pan_angle'] = angle
                depth = target.get('depth_cm', -1)
                log.info("✓ QR CODE FOUND at pan=%d° depth≈%.0fcm → APPROACH_QR",
                         angle, depth)
                return STATE_APPROACH_QR, target

        self.scan_index += 1
        return current_state, None
