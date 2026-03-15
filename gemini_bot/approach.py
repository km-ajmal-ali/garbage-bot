"""
GeminiBot Approach & Collect Module
=====================================
Handles the APPROACH state — aligns the robot to the detected garbage
angle, then drives toward it while periodically sending images to
Gemini for guidance.  Once Gemini reports the object is "very_close"
or "collected", the robot enters the final collection push and
returns to SCANNING.

Cost optimisation:
  • Images are sent at a configurable interval (default 2s)
  • Between Gemini calls the robot drives blind using the last instruction
  • Images are resized + compressed before sending
"""

import time

from core.logger import get_logger
from gemini_bot.config import (
    APPROACH_SPEED, COLLECT_SPEED,
    APPROACH_IMAGE_INTERVAL,
    APPROACH_DRIVE_PULSE, APPROACH_STEER_PULSE,
    MAX_APPROACH_CYCLES, MAX_LOST_ATTEMPTS,
    PAN_CENTER_ANGLE, TILT_CENTER_ANGLE,
)
from gemini_bot.camera import read_frame, flush_camera_buffer
from gemini_bot.gemini_vision import approach_image

log = get_logger("g_approach")


class GeminiApproacher:
    """
    Drives the robot toward a detected waste object, using Gemini
    for periodic visual guidance.
    """

    def __init__(self, camera, motors, pan_servo, tilt_servo):
        self.camera     = camera
        self.motors     = motors
        self.pan_servo  = pan_servo
        self.tilt_servo = tilt_servo
        log.info("GeminiApproacher initialised (speed=%d%%, interval=%.1fs, max_cycles=%d)",
                 APPROACH_SPEED, APPROACH_IMAGE_INTERVAL, MAX_APPROACH_CYCLES)

    def execute(self, target_info: dict, running_flag: callable) -> str:
        """
        Align to the target angle, then drive toward the object with
        periodic Gemini guidance.

        Args:
            target_info:  dict from scanner with 'angle', 'objects', 'best'
            running_flag: callable returning True while the bot should run

        Returns:
            Next state string: "SCANNING"
        """
        target_angle = target_info.get("angle", 0)
        best_obj = target_info.get("best", {})
        log.info("═══ APPROACH started  target_angle=%d°  object='%s' ═══",
                 target_angle, best_obj.get("label", "?"))

        # ── Step 1: Align chassis to the scan angle ──────────────────
        # Centre the pan servo first (camera looks forward)
        log.info("Centring pan servo for forward view")
        self.pan_servo.move_and_detach(PAN_CENTER_ANGLE, settle=0.2)

        # If the object was not at centre, rotate the chassis to face it
        if target_angle != 0:
            direction = "right" if target_angle > 0 else "left"
            # Rough conversion: rotate proportional to angle offset
            rotate_time = abs(target_angle) / 90.0 * 1.2  # ~1.2s for full 90°
            log.info("Rotating chassis %s for %.2fs to align to %d°",
                     direction, rotate_time, target_angle)
            self.motors.move(direction, APPROACH_SPEED)
            time.sleep(rotate_time)
            self.motors.stop()
            time.sleep(0.3)

        # ── Step 2: Approach loop with periodic Gemini guidance ───────
        lost_count = 0
        cycle = 0

        while running_flag() and cycle < MAX_APPROACH_CYCLES:
            cycle += 1
            log.info("─── Approach cycle %d/%d ───", cycle, MAX_APPROACH_CYCLES)

            # Capture a fresh frame
            flush_camera_buffer(self.camera, 2)
            ret, frame = read_frame(self.camera)
            if not ret or frame is None:
                log.warning("Camera read failed in approach cycle %d", cycle)
                time.sleep(0.5)
                continue

            # ── Ask Gemini for guidance ────────────────────────────────
            result = approach_image(frame)

            obj_visible = result.get("object_visible", False)
            action      = result.get("action", "stop")
            distance    = result.get("distance", "unknown")
            collected   = result.get("collected", False)
            position    = result.get("position", "center")

            log.info("Gemini guidance: visible=%s  action=%s  dist=%s  pos=%s  collected=%s",
                     obj_visible, action, distance, position, collected)

            # ── Check if collection is done ────────────────────────────
            if collected:
                log.info("✓ Gemini says COLLECTED! Running final collection push …")
                self._collection_push(running_flag)
                return "SCANNING"

            if not obj_visible:
                lost_count += 1
                log.warning("Object not visible (lost_count=%d/%d)",
                            lost_count, MAX_LOST_ATTEMPTS)
                if lost_count >= MAX_LOST_ATTEMPTS:
                    log.warning("Object lost for %d cycles → back to SCANNING",
                                MAX_LOST_ATTEMPTS)
                    self.motors.stop()
                    self._reset_servos()
                    return "SCANNING"
                # Drive forward a bit hoping it comes back into view
                self.motors.move("forward", APPROACH_SPEED)
                time.sleep(APPROACH_DRIVE_PULSE)
                self.motors.stop()
                time.sleep(APPROACH_IMAGE_INTERVAL)
                continue

            lost_count = 0

            # ── Execute the action ────────────────────────────────────
            if action == "stop" or distance == "very_close":
                log.info("Object very close → final collection push")
                self._collection_push(running_flag)
                return "SCANNING"

            elif action == "left":
                log.info("Steering LEFT")
                self.motors.move("left", APPROACH_SPEED)
                time.sleep(APPROACH_STEER_PULSE)
                self.motors.stop()

            elif action == "right":
                log.info("Steering RIGHT")
                self.motors.move("right", APPROACH_SPEED)
                time.sleep(APPROACH_STEER_PULSE)
                self.motors.stop()

            elif action == "forward":
                log.info("Driving FORWARD (dist=%s)", distance)
                # Drive longer for far objects, shorter for close
                pulse = APPROACH_DRIVE_PULSE
                if distance == "far":
                    pulse *= 2.0
                elif distance == "close":
                    pulse *= 0.5
                self.motors.move("forward", APPROACH_SPEED)
                time.sleep(pulse)
                self.motors.stop()

            else:
                log.warning("Unknown action '%s' – stopping", action)
                self.motors.stop()

            # ── Wait before next Gemini call (cost control) ───────────
            log.debug("Waiting %.1fs before next Gemini call", APPROACH_IMAGE_INTERVAL)
            time.sleep(APPROACH_IMAGE_INTERVAL)

        # ── Max cycles exhausted ──────────────────────────────────────
        log.warning("Max approach cycles (%d) reached → SCANNING", MAX_APPROACH_CYCLES)
        self.motors.stop()
        self._reset_servos()
        return "SCANNING"

    def _collection_push(self, running_flag: callable):
        """
        Final forward drive to roll over / collect the object.
        Drives forward for a fixed duration, then verifies with Gemini.
        """
        log.info("═══ COLLECTION PUSH ═══")
        log.info("Driving forward at %d%% for 2 seconds", COLLECT_SPEED)
        self.motors.move("forward", COLLECT_SPEED)
        time.sleep(2.0)
        self.motors.stop()

        # Verify collection with one more Gemini call
        ret, frame = read_frame(self.camera)
        if ret and frame is not None:
            result = approach_image(frame)
            if result.get("object_visible", False) and not result.get("collected", False):
                log.info("Object still visible after push — one more forward burst")
                self.motors.move("forward", COLLECT_SPEED)
                time.sleep(1.5)
                self.motors.stop()

        log.info("✓ Collection push complete")
        self._reset_servos()

    def _reset_servos(self):
        """Centre both servos for the next scan cycle."""
        log.info("Resetting servos to centre")
        self.pan_servo.move_and_detach(PAN_CENTER_ANGLE, settle=0.15)
        self.tilt_servo.move_and_detach(TILT_CENTER_ANGLE, settle=0.15)
