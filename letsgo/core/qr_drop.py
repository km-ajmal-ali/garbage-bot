"""
WasteBot QR Drop Module
========================
Handles the DROP_QR state – dropping the waste and turning around.
"""

import time

from core.config import (
    COLLECT_SPEED,
    CHASSIS_TURN_SPEED,
    MAX_OPEN_ANGLE,
    CHASSIS_DEGREES_PER_SEC
)
from core.states import STATE_SEARCH_ROTATE
from core.logger import get_logger

log = get_logger("qr_drop")

class QRDrop:
    def __init__(self, motors, collector_servo, display):
        self.motors = motors
        self.collector_servo = collector_servo
        self.display = display

    def execute(self, running_flag: callable) -> str:
        log.info("═══ DROP_QR started ═══")
        
        # 1. Drive forward slightly to push garbage in
        if running_flag():
            log.info("Driving forward softly to push waste")
            self.motors.move("forward", COLLECT_SPEED)
            time.sleep(1.0)
            self.motors.stop()

        # 2. Open gripper
        if running_flag():
            log.info("Opening gripper to release waste")
            self.collector_servo.move_and_detach(MAX_OPEN_ANGLE, settle=1.0)

        # 3. Reverse to clear the waste
        if running_flag():
            log.info("Reversing to clear the waste")
            self.motors.move("backward", COLLECT_SPEED)
            time.sleep(1.5)
            self.motors.stop()

        # 4. Turn 180 degrees
        if running_flag():
            log.info("Turning 180 degrees to search for new garbage")
            turn_time = 180 / CHASSIS_DEGREES_PER_SEC
            self.motors.move("right", CHASSIS_TURN_SPEED)
            time.sleep(turn_time)
            self.motors.stop()

        log.info("═══ DROP_QR done → SEARCH_ROTATE ═══")
        return STATE_SEARCH_ROTATE
