"""
WasteBot Collector Module
==========================
Handles the COLLECT state – moving forward to enclose, grip, push and release objects.
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
        log.info("Collector initialised (speed=%d%%)", COLLECT_SPEED)

    def execute(self, running_flag: callable) -> str:
        """
        Executes the timed collector sequence.
        (Visual loop is replaced with timed movement because gripping happens 
        at a rigid physical distance setup by the APPROACH logic.)
        """
        log.info("═══ COLLECT started ═══")
        log.info("Ensuring gripper is open")
        self.collector_servo.move_and_detach(MAX_OPEN_ANGLE, settle=0.3)
        
        def pump_frames(duration, text, direction=None, speed=COLLECT_SPEED):
            """Helper to keep camera window alive during timed actions."""
            if direction:
                self.motors.move(direction, speed)
            start_time = time.time()
            frame_num = 0
            while running_flag() and (time.time() - start_time < duration):
                ret, frame = read_frame(self.camera)
                if ret and frame is not None:
                    # Show empty detections explicitly so UI updates immediately
                    self.display.show(frame, [], text)
                frame_num += 1
            if direction:
                self.motors.stop()
            return frame_num

        # Move forward slightly to put the bottle inside the physical gripper jaws
        log.info("Driving FORWARD (0.8s) to enclose object")
        frames_enc = pump_frames(0.8, "COLLECT (ENCLOSING)", direction="forward")

        if not running_flag():
            return STATE_SCANNING

        # Wait a moment before clamping
        time.sleep(0.5)

        log.info("Closing gripper around object")
        self.collector_servo.move_and_detach(MAX_CLOSE_ANGLE, settle=1.0)
        
        log.info("Driving forward softly with object (1.5s)")
        frames_push = pump_frames(1.5, "COLLECT (PUSHING)", direction="forward")

        time.sleep(0.5)

        log.info("Opening gripper to release")
        self.collector_servo.move_and_detach(MAX_OPEN_ANGLE, settle=1.0)

        log.info("Moving backward slightly before search rotation (1.0s)")
        frames_rev = pump_frames(1.0, "COLLECT (REVERSING)", direction="backward")

        # ── Reset hardware ────────────────────────────────────────────
        log.info("Resetting servos to centre")
        self.tilt_servo.move_and_detach(TILT_CENTER_ANGLE, settle=0.15)
        self.pan_servo.move_and_detach(PAN_CENTER_ANGLE, settle=0.15)

        total_frames = frames_enc + frames_push + frames_rev
        log.info("═══ COLLECT done (pumped %d frames) → SEARCH_ROTATE ═══\n", total_frames)
        return STATE_SEARCH_ROTATE
