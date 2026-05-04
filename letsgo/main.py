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
from core.config import (
    MOTOR_PINS, PAN_SERVO_PIN, TILT_SERVO_PIN, COLLECTOR_SERVO_PIN,
    PAN_MIN_ANGLE, PAN_MAX_ANGLE, PAN_CENTER_ANGLE,
    TILT_MIN_ANGLE, TILT_MAX_ANGLE, TILT_CENTER_ANGLE,
    MAX_OPEN_ANGLE, MAX_CLOSE_ANGLE,
    SCAN_DWELL, CAMERA_ROTATE_180,
    CHASSIS_TURN_SPEED, CHASSIS_DEGREES_PER_SEC, PAN_ALIGN_THRESHOLD,
)
from core.states import (
    STATE_SCANNING,
    STATE_APPROACH,
    STATE_TILT_ADJUST,
    STATE_COLLECT,
    STATE_SEARCH_ROTATE,
    STATE_SEARCH_QR,
    STATE_SEARCH_ROTATE_QR,
    STATE_APPROACH_QR,
    STATE_DROP_QR,
)
from core.detection import load_model, init_camera, read_frame
from core.display    import Display
from core.scanner    import Scanner
from core.approach   import Approacher
from core.tilt_adjust import TiltAdjuster
from core.collector  import Collector
from core.navigator  import Navigator
from core.qr_scanner import QRScanner
from core.qr_navigator import QRNavigator
from core.qr_approach import QRApproacher
from core.qr_drop import QRDrop

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

        # ── Camera ────────────────────────────────────────────────────
        self.camera = init_camera()
        if self.camera is None:
            log.error("Could not open any camera backend!")
            raise RuntimeError("Could not open any camera backend.")

        # ── AI model ──────────────────────────────────────────────────
        self.model = load_model()

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

        log.info("Initialising collector servo on GPIO %d  limits=[%d°, %d°]  center=%d°",
                 COLLECTOR_SERVO_PIN, MAX_CLOSE_ANGLE, MAX_OPEN_ANGLE, MAX_OPEN_ANGLE)
        self.collector_servo = CameraServo(
            pin=COLLECTOR_SERVO_PIN,
            min_limit=MAX_CLOSE_ANGLE,
            max_limit=MAX_OPEN_ANGLE,
            center_angle=MAX_OPEN_ANGLE,
        )

        # Centre servos on startup
        log.info("Centring all servos (starting gripper in open state)")
        self.pan_servo.center()
        self.tilt_servo.center()
        self.collector_servo.center()

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
                                     self.tilt_servo, self.collector_servo, self.display)
        self.navigator   = Navigator(self.model, self.camera,
                                     self.motors, self.display)

        self.qr_scanner  = QRScanner(self.camera, self.pan_servo, self.display)
        self.qr_navigator = QRNavigator(self.camera, self.motors, self.display)
        self.qr_approacher = QRApproacher(self.camera, self.motors, self.display)
        self.qr_drop     = QRDrop(self.motors, self.collector_servo, self.display)

        # ── Runtime state ─────────────────────────────────────────────
        self.state   = STATE_SCANNING
        self.target  = None
        self.running = True

        log.info("WasteBot ready.  Initial state → %s\n", self.state)

    # ──────────────────────────────────────────────────────────────────
    # Chassis alignment (rotate body to face scanned target)
    # ──────────────────────────────────────────────────────────────────
    def _align_chassis_to_pan(self, pan_angle: float):
        """
        Rotate the chassis so it faces the direction the camera was
        pointing when the scanner found the target, then centre the
        pan servo so the approach module can steer normally.

        Args:
            pan_angle: Pan servo angle (degrees) where the target was found.
        """
        offset = abs(pan_angle - PAN_CENTER_ANGLE)
        if offset <= PAN_ALIGN_THRESHOLD:
            log.info("Pan offset %d° ≤ threshold %d° – skipping chassis alignment",
                     offset, PAN_ALIGN_THRESHOLD)
            return

        # Calculate rotation time from calibrated rate
        turn_time = offset / CHASSIS_DEGREES_PER_SEC

        # Determine turn direction.
        # When the camera is upside-down the servo's physical pan
        # direction is inverted relative to the corrected image,
        # so we flip the motor direction.
        if pan_angle > PAN_CENTER_ANGLE:
            direction = "left" if CAMERA_ROTATE_180 else "right"
        else:
            direction = "right" if CAMERA_ROTATE_180 else "left"

        log.info("╔══ CHASSIS ALIGN: pan=%d° → rotating %s for %.2fs (%.0f°/s) ══╗",
                 pan_angle, direction.upper(), turn_time, CHASSIS_DEGREES_PER_SEC)

        self.motors.move(direction, CHASSIS_TURN_SPEED)
        time.sleep(turn_time)
        self.motors.stop()
        time.sleep(0.2)   # let momentum settle

        # Centre the pan servo so the approach module starts with a
        # forward-facing camera
        log.info("Centring pan servo → %d°", PAN_CENTER_ANGLE)
        self.pan_servo.move_and_detach(PAN_CENTER_ANGLE, settle=SCAN_DWELL)

        # Flush stale frames captured during the turn
        log.debug("Flushing camera buffer after chassis alignment")
        for _ in range(3):
            read_frame(self.camera)

        log.info("╚══ CHASSIS ALIGN complete ══╝")

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

        log.info("Centring servos (opening gripper)")
        self.pan_servo.center()
        self.tilt_servo.center()
        self.collector_servo.center()
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
                        # Rotate chassis to face the direction the camera
                        # was pointing before entering APPROACH
                        pan_angle = target.get('pan_angle', PAN_CENTER_ANGLE)
                        self._align_chassis_to_pan(pan_angle)
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
                    self.navigator.reset()

                # ── SEARCH ROTATE ─────────────────────────────────────
                elif self.state == STATE_SEARCH_ROTATE:
                    next_state, target = self.navigator.step()
                    if target:
                        self.target = target
                    self.state = next_state
                    if next_state == STATE_SCANNING:
                        self.scanner.reset()

                # ── SEARCH QR ─────────────────────────────────────────
                elif self.state == STATE_SEARCH_QR:
                    next_state, target = self.qr_scanner.step(self.state)
                    if target:
                        self.target = target
                        self.qr_navigator.reset()
                        pan_angle = target.get('pan_angle', PAN_CENTER_ANGLE)
                        self._align_chassis_to_pan(pan_angle)
                    self.state = next_state

                # ── SEARCH ROTATE QR ──────────────────────────────────
                elif self.state == STATE_SEARCH_ROTATE_QR:
                    next_state, target = self.qr_navigator.step()
                    if target:
                        self.target = target
                    self.state = next_state
                    if next_state == STATE_SEARCH_QR:
                        self.qr_scanner.reset()

                # ── APPROACH QR ───────────────────────────────────────
                elif self.state == STATE_APPROACH_QR:
                    next_state, target = self.qr_approacher.execute(
                        running_flag=lambda: self.running
                    )
                    if target:
                        self.target = target
                    self.state = next_state
                    if next_state == STATE_SEARCH_QR:
                        self.qr_scanner.reset()

                # ── DROP QR ───────────────────────────────────────────
                elif self.state == STATE_DROP_QR:
                    self.state = self.qr_drop.execute(
                        running_flag=lambda: self.running
                    )
                    self.qr_scanner.reset()
                    self.qr_navigator.reset()

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
