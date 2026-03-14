"""
GeminiBot – Autonomous Garbage Collection Robot (Gemini Vision)
================================================================

Main entry point.  Uses Google Gemini API for vision-based garbage
detection instead of Hailo on-device inference.

Hardware: Raspberry Pi 5 • 2× 500 RPM geared motors
          2× camera-gimbal servos • IMX219 CSI camera

State Machine:
    SCANNING      → sweep pan servo, capture images, ask Gemini if garbage
    APPROACH      → align + drive toward object, periodic Gemini guidance
    SEARCH_ROTATE → rotate chassis if scan sweep found nothing

Usage:
    python3 -m gemini_bot.main
    python3 -m gemini_bot.main --log=INFO
    python3 -m gemini_bot.main --log=DEBUG
"""

import sys
import os
import signal
import time

# ── Ensure project root is on the path ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

# ── Logging (set up BEFORE any other imports) ─────────────────────────
from core.logger import setup_logging, get_logger

_log_level = "DEBUG"
for arg in sys.argv[1:]:
    if arg.startswith("--log="):
        _log_level = arg.split("=", 1)[1].upper()
setup_logging(level=_log_level)

log = get_logger("gemini_main")

# ── Configuration ─────────────────────────────────────────────────────
from gemini_bot.config import (
    MOTOR_PINS, PAN_SERVO_PIN, TILT_SERVO_PIN,
    PAN_MIN_ANGLE, PAN_MAX_ANGLE, PAN_CENTER_ANGLE,
    TILT_MIN_ANGLE, TILT_MAX_ANGLE, TILT_CENTER_ANGLE,
)

# ── Hardware drivers (shared with letsgo) ─────────────────────────────
from common.motors import MotorControl
from common.servos import CameraServo

# ── GeminiBot modules ─────────────────────────────────────────────────
from gemini_bot.camera    import init_camera
from gemini_bot.scanner   import GeminiScanner
from gemini_bot.approach  import GeminiApproacher
from gemini_bot.navigator import GeminiNavigator


# ═══════════════════════════════════════════════════════════════════════
# GeminiBot Controller
# ═══════════════════════════════════════════════════════════════════════

class GeminiBot:
    """
    Top-level controller.
    Wires up hardware + Gemini-based modules and runs the state machine.
    """

    # ── States ────────────────────────────────────────────────────────
    STATE_SCANNING      = "SCANNING"
    STATE_APPROACH      = "APPROACH"
    STATE_SEARCH_ROTATE = "SEARCH_ROTATE"

    def __init__(self):
        log.info("=" * 60)
        log.info("  GeminiBot – Garbage Collector (Gemini Vision)")
        log.info("=" * 60)
        log.info("Log level: %s", _log_level)

        # ── Camera ────────────────────────────────────────────────────
        self.camera = init_camera()
        if self.camera is None:
            log.error("Could not open any camera backend!")
            raise RuntimeError("Could not open any camera backend.")

        # ── Motors ────────────────────────────────────────────────────
        log.info("Initialising motors on pins %s", MOTOR_PINS)
        self.motors = MotorControl(MOTOR_PINS)

        # ── Servos (pan = X, tilt = Y) ───────────────────────────────
        log.info("Initialising pan servo (X-axis) on GPIO %d  limits=[%d°, %d°]  center=%d°",
                 PAN_SERVO_PIN, PAN_MIN_ANGLE, PAN_MAX_ANGLE, PAN_CENTER_ANGLE)
        self.pan_servo = CameraServo(
            pin=PAN_SERVO_PIN,
            min_limit=PAN_MIN_ANGLE,
            max_limit=PAN_MAX_ANGLE,
            center_angle=PAN_CENTER_ANGLE,
        )

        log.info("Initialising tilt servo (Y-axis) on GPIO %d  limits=[%d°, %d°]  center=%d°",
                 TILT_SERVO_PIN, TILT_MIN_ANGLE, TILT_MAX_ANGLE, TILT_CENTER_ANGLE)
        self.tilt_servo = CameraServo(
            pin=TILT_SERVO_PIN,
            min_limit=TILT_MIN_ANGLE,
            max_limit=TILT_MAX_ANGLE,
            center_angle=TILT_CENTER_ANGLE,
        )

        # Centre servos on startup
        log.info("Centring both servos")
        self.pan_servo.center()
        self.tilt_servo.center()

        # ── State-machine modules ────────────────────────────────────
        self.scanner   = GeminiScanner(self.camera, self.pan_servo)
        self.approacher = GeminiApproacher(
            self.camera, self.motors, self.pan_servo, self.tilt_servo
        )
        self.navigator = GeminiNavigator(self.camera, self.motors)

        # ── Runtime state ─────────────────────────────────────────────
        self.state   = self.STATE_SCANNING
        self.target  = None
        self.running = True

        log.info("GeminiBot ready.  Initial state → %s\n", self.state)

    # ──────────────────────────────────────────────────────────────────
    # Graceful shutdown
    # ──────────────────────────────────────────────────────────────────
    def shutdown(self, signum=None, frame=None):
        """Stop all hardware and release resources."""
        if getattr(self, '_shutdown_executed', False):
            return
        self._shutdown_executed = True

        log.info("SHUTDOWN signal received")
        self.running = False

        log.info("Stopping motors")
        self.motors.stop()

        log.info("Centring servos")
        self.pan_servo.center()
        self.tilt_servo.center()
        time.sleep(0.3)

        log.info("Releasing camera")
        self.camera.release()

        log.info("Cleaning up motor GPIO")
        self.motors.cleanup()

        log.info("SHUTDOWN complete ✓")

    # ──────────────────────────────────────────────────────────────────
    # State-machine loop
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        """
        Main loop – dispatches to the active state handler.
        Press Ctrl-C to exit gracefully.
        """
        def handle_sigterm(signum, frame):
            log.info("SIGTERM received, raising KeyboardInterrupt")
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, handle_sigterm)

        log.info("🤖  GeminiBot is awake!  Beginning scan …\n")

        try:
            loop_count = 0
            while self.running:
                loop_count += 1
                prev_state = self.state

                # ── SCANNING ──────────────────────────────────────────
                if self.state == self.STATE_SCANNING:
                    next_state, target = self.scanner.step()
                    if target:
                        self.target = target
                        self.navigator.reset()
                    self.state = next_state

                # ── APPROACH ──────────────────────────────────────────
                elif self.state == self.STATE_APPROACH:
                    if self.target is None:
                        log.warning("APPROACH with no target → SCANNING")
                        self.state = self.STATE_SCANNING
                        continue

                    next_state = self.approacher.execute(
                        target_info=self.target,
                        running_flag=lambda: self.running,
                    )
                    self.state = next_state
                    self.target = None
                    self.scanner.reset()

                # ── SEARCH ROTATE ─────────────────────────────────────
                elif self.state == self.STATE_SEARCH_ROTATE:
                    next_state, target = self.navigator.step()
                    if target:
                        self.target = target
                    self.state = next_state
                    if next_state == self.STATE_SCANNING:
                        self.scanner.reset()

                else:
                    log.error("Unknown state '%s' → SCANNING", self.state)
                    self.state = self.STATE_SCANNING

                # Log state transitions
                if self.state != prev_state:
                    log.info("━━━ STATE TRANSITION: %s → %s  (loop #%d) ━━━",
                             prev_state, self.state, loop_count)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received")
        finally:
            self.shutdown()


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = GeminiBot()
    bot.run()
