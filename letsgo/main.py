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
"""

import sys
import os
import signal
import time

# ── Ensure project root is on the path ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Core modules ──────────────────────────────────────────────────────────
from core.config import MOTOR_PINS, PAN_SERVO_PIN, TILT_SERVO_PIN
from core.states import (
    STATE_SCANNING,
    STATE_APPROACH,
    STATE_TILT_ADJUST,
    STATE_COLLECT,
    STATE_SEARCH_ROTATE,
)
from core.detection import load_model, init_camera
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
        print("=" * 60)
        print("  WasteBot – Autonomous Garbage Collector")
        print("=" * 60)

        # ── AI model ──────────────────────────────────────────────────
        self.model = load_model()

        # ── Camera ────────────────────────────────────────────────────
        self.camera = init_camera()
        if self.camera is None:
            raise RuntimeError("Could not open any camera backend.")

        # ── Motors ────────────────────────────────────────────────────
        print("[INIT] Initialising motors …")
        self.motors = MotorControl(MOTOR_PINS)

        # ── Servos (pan = X, tilt = Y) ───────────────────────────────
        print("[INIT] Initialising pan servo (X-axis) …")
        self.pan_servo = CameraServo(pin=PAN_SERVO_PIN)

        print("[INIT] Initialising tilt servo (Y-axis) …")
        self.tilt_servo = CameraServo(pin=TILT_SERVO_PIN)

        # Centre servos on startup
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

        print("[INIT] WasteBot ready.  State → SCANNING\n")

    # ──────────────────────────────────────────────────────────────────
    # Graceful shutdown
    # ──────────────────────────────────────────────────────────────────
    def shutdown(self, signum=None, frame=None):
        """Stop all hardware and release resources."""
        print("\n[SHUTDOWN] Cleaning up …")
        self.running = False
        self.motors.stop()
        self.pan_servo.center()
        self.tilt_servo.center()
        time.sleep(0.3)
        self.camera.release()
        self.motors.cleanup()
        print("[SHUTDOWN] Done.")

    # ──────────────────────────────────────────────────────────────────
    # State-machine loop
    # ──────────────────────────────────────────────────────────────────
    def run(self):
        """
        Main loop – dispatches to the active state handler.
        Press Ctrl-C to exit gracefully.
        """
        signal.signal(signal.SIGINT,  self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        print("🤖  WasteBot is awake!  Beginning scan …\n")

        try:
            while self.running:

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
                    print(f"[ERROR] Unknown state '{self.state}' → resetting.")
                    self.state = STATE_SCANNING

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = WasteBot()
    bot.run()
