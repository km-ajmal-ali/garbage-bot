"""
WasteBot – Autonomous Garbage Collection Robot
================================================

Main entry point.  Initialises all hardware and modules, then runs
the finite-state machine loop.

Hardware: Raspberry Pi 5 • Hailo AI HAT+ (26 TOPS)
          2× 500 RPM geared motors  •  2× camera-gimbal servos
          IMX219 CSI camera

Usage:
    python3 main.py
    python3 main.py --log=INFO      # less verbose
    python3 main.py --log=DEBUG     # full diagnostic output (default)
"""

import sys
import os
import signal
import time

# ── Ensure project root is on the path ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Logging (must be set up BEFORE any other core imports) ────────────────
from core.logger import setup_logging, get_logger

# Parse --log=LEVEL from command line
_log_level = "DEBUG"
for arg in sys.argv[1:]:
    if arg.startswith("--log="):
        _log_level = arg.split("=", 1)[1].upper()
setup_logging(level=_log_level)

log = get_logger("main")

# ── Core modules ──────────────────────────────────────────────────────────
from core.config import MOTOR_PINS, PAN_SERVO_PIN, TILT_SERVO_PIN
from core.states import (
    STATE_SCANNING,
    STATE_APPROACH,
    STATE_TILT_ADJUST,
    STATE_COLLECT,
    STATE_SEARCH_ROTATE,
)
from core.detection import load_model, init_camera, run_detection
import numpy as np
from core.display    import Display
from core.scanner    import Scanner
from core.approach   import Approacher
from core.tilt_adjust import TiltAdjuster
from core.collector  import Collector
from core.navigator  import Navigator

# ── Hardware drivers ──────────────────────────────────────────────────────
from common.motors  import MotorControl
from common.servos  import CameraServo


# ═══════════════════════════════════════════════════════════════════════════
# WasteBot Controller
# ═══════════════════════════════════════════════════════════════════════════

class WasteBot:
    """
    Top-level controller.
    Wires up hardware + modules and runs the state-machine loop.
    """

    def __init__(self):
        log.info("=" * 60)
        log.info("  WasteBot – Autonomous Garbage Collector")
        log.info("=" * 60)
        log.info("Log level: %s", _log_level)

        # ── AI model ──────────────────────────────────────────────────
        self.model = load_model()

        # Warm up the Hailo AI chip with a dummy inference.
        # This prevents a massive power-surge from Hailo compiling
        # its first pipeline concurrently with the servos moving.
        log.info("Warming up Hailo AI model with dummy frame...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        run_detection(self.model, dummy_frame)
        log.info("AI warm-up complete.")

        # ── Camera ────────────────────────────────────────────────────
        self.camera = init_camera()
        if self.camera is None:
            log.error("Could not open any camera backend!")
            raise RuntimeError("Could not open any camera backend.")

        # ── Motors ────────────────────────────────────────────────────
        log.info("Initialising motors on pins %s", MOTOR_PINS)
        self.motors = MotorControl(MOTOR_PINS)

        # ── Servos (pan = X, tilt = Y) ───────────────────────────────
        log.info("Initialising pan servo (X-axis) on GPIO %d", PAN_SERVO_PIN)
        self.pan_servo = CameraServo(pin=PAN_SERVO_PIN)

        log.info("Initialising tilt servo (Y-axis) on GPIO %d", TILT_SERVO_PIN)
        self.tilt_servo = CameraServo(pin=TILT_SERVO_PIN)

        # Centre servos on startup
        log.info("Centring both servos")
        self.pan_servo.center()
        self.tilt_servo.center()

        # ── Display ───────────────────────────────────────────────────
        self.display = Display()

        # ── State-machine modules ─────────────────────────────────────
        self.scanner     = Scanner(self.model, self.camera,
                                   self.pan_servo, self.display)
        self.approacher  = Approacher(self.model, self.camera,
                                      self.motors, self.display)
        self.tilt_adj    = TiltAdjuster(self.model, self.camera,
                                        self.tilt_servo, self.display)
        self.collector   = Collector(self.model, self.camera,
                                     self.motors, self.pan_servo,
                                     self.tilt_servo, self.display)
        self.navigator   = Navigator(self.model, self.camera,
                                     self.motors, self.display)

        # ── Runtime state ─────────────────────────────────────────────
        self.state   = STATE_SCANNING
        self.target  = None
        self.running = True

        log.info("WasteBot ready.  Initial state → %s\n", self.state)

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

        # Close Hailo pipeline if model supports it
        if hasattr(self.model, 'close'):
            log.info("Closing Hailo inference pipeline")
            self.model.close()

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

        # By NOT overriding SIGINT, python natively raises KeyboardInterrupt on Ctrl+C!
        signal.signal(signal.SIGTERM, handle_sigterm)

        log.info("🤖  WasteBot is awake!  Beginning scan …\n")

        try:
            loop_count = 0
            while self.running:
                loop_count += 1
                prev_state = self.state

                # ── SCANNING ──────────────────────────────────────────
                if self.state == STATE_SCANNING:
                    next_state, target = self.scanner.step(self.state)
                    if target:
                        self.target = target
                        self.navigator.reset()
                    self.state = next_state

                # ── APPROACH ──────────────────────────────────────────
                elif self.state == STATE_APPROACH:
                    next_state, target = self.approacher.execute(
                        running_flag=lambda: self.running
                    )
                    if target:
                        self.target = target
                    self.state = next_state
                    if next_state == STATE_SCANNING:
                        self.scanner.reset()

                # ── TILT ADJUST ───────────────────────────────────────
                elif self.state == STATE_TILT_ADJUST:
                    next_state, target = self.tilt_adj.execute(
                        running_flag=lambda: self.running
                    )
                    if target:
                        self.target = target
                    self.state = next_state

                # ── COLLECT ───────────────────────────────────────────
                elif self.state == STATE_COLLECT:
                    self.state = self.collector.execute(
                        running_flag=lambda: self.running
                    )
                    self.scanner.reset()

                # ── SEARCH ROTATE ─────────────────────────────────────
                elif self.state == STATE_SEARCH_ROTATE:
                    next_state, target = self.navigator.step()
                    if target:
                        self.target = target
                    self.state = next_state
                    if next_state == STATE_SCANNING:
                        self.scanner.reset()

                else:
                    log.error("Unknown state '%s' → resetting to SCANNING", self.state)
                    self.state = STATE_SCANNING

                # Log state transitions
                if self.state != prev_state:
                    log.info("━━━ STATE TRANSITION: %s → %s  (loop #%d) ━━━",
                             prev_state, self.state, loop_count)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received")
        finally:
            self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = WasteBot()
    bot.run()
