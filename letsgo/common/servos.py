from gpiozero import AngularServo
from time import sleep
from core.logger import get_logger

log = get_logger("servo")

# Attempt to use pigpio factory for smoother servo movement
try:
    from gpiozero.pins.pigpio import PiGPIOFactory
    from gpiozero import Device
    # Only set if not already set (avoid re-initialization)
    if Device.pin_factory is None or not isinstance(Device.pin_factory, PiGPIOFactory):
        log.info("Initializing pigpio factory for smooth servo control")
        Device.pin_factory = PiGPIOFactory()
        log.info("pigpio factory active")
except ImportError:
    log.warning("pigpio python library not found. Install with 'sudo apt install python3-pigpio'")
except Exception as e:
    log.warning("Could not connect to pigpio daemon: %s. Servos may jitter.", e)
    log.warning("Ensure pigpiod is running: 'sudo pigpiod'")


class CameraServo:
    def __init__(self, pin=25, min_angle=-90, max_angle=90,
                 min_limit=None, max_limit=None, center_angle=0):
        """
        Initializes a servo for camera control.

        Args:
            pin:          GPIO pin number
            min_angle:    AngularServo minimum angle (hardware range)
            max_angle:    AngularServo maximum angle (hardware range)
            min_limit:    Soft minimum angle limit to prevent over-rotation.
                          Defaults to min_angle if not specified.
            max_limit:    Soft maximum angle limit to prevent over-rotation.
                          Defaults to max_angle if not specified.
            center_angle: The neutral / home position angle.
        """
        self.pin = pin
        self.min_limit = min_limit if min_limit is not None else min_angle
        self.max_limit = max_limit if max_limit is not None else max_angle
        self.center_angle = center_angle

        log.info("Init servo on GPIO %d  range=[%d°, %d°]  limits=[%d°, %d°]  center=%d°",
                 pin, min_angle, max_angle, self.min_limit, self.max_limit, self.center_angle)

        self.servo = AngularServo(pin,
                                  min_angle=min_angle,
                                  max_angle=max_angle,
                                  min_pulse_width=0.0005,
                                  max_pulse_width=0.0025,
                                  initial_angle=None)
        self.current_angle = self.center_angle
        log.info("Servo GPIO %d initialised (PWM off)", pin)

    def clamp_angle(self, angle):
        """Clamp an angle to the configured [min_limit, max_limit] range."""
        clamped = max(self.min_limit, min(self.max_limit, angle))
        if clamped != angle:
            log.warning("Servo GPIO %d: angle %d° clamped to %d° (limits=[%d°, %d°])",
                        self.pin, angle, clamped, self.min_limit, self.max_limit)
        return clamped

    def set_angle(self, angle):
        """Sets the servo to a specific angle (clamped to limits, includes settle sleep, then detaches)."""
        angle = self.clamp_angle(angle)
        self.servo.angle = angle
        self.current_angle = angle
        log.debug("Servo GPIO %d → %d°  (settling 0.5s)", self.pin, angle)
        sleep(0.5)  # Give time for physical movement
        self.servo.detach()  # Turn off PWM signal to prevent jittering/buzzing

    def move_and_detach(self, angle, settle=0.25):
        """
        Move to a clamped angle, wait for it to settle, then detach PWM.

        Use this instead of writing to self.servo.angle directly —
        it prevents jitter/vibration by turning off the PWM signal
        once the servo has reached its target position.

        Args:
            angle:  Target angle in degrees (will be clamped to limits).
            settle: Seconds to wait for the servo to physically settle
                    before detaching PWM.  Shorter = faster sweep,
                    longer = more reliable positioning.
        """
        angle = self.clamp_angle(angle)
        self.servo.angle = angle
        self.current_angle = angle
        sleep(settle)
        self.servo.detach()
        log.debug("Servo GPIO %d → %d°  (settled %.2fs, PWM off)", self.pin, angle, settle)

    def look_around(self):
        """
        A routine for the robot to scan its environment.
        Moves the camera in steps within the configured limits.
        """
        log.info("Servo GPIO %d: look_around sweep starting", self.pin)
        positions = [self.center_angle, 45, 90, 45, self.center_angle, -45, -90, -45, self.center_angle]
        for pos in positions:
            self.set_angle(pos)  # set_angle already clamps
            sleep(0.5)
        log.info("Servo GPIO %d: look_around sweep complete", self.pin)

    def center(self):
        """Resets camera to the configured center angle."""
        log.debug("Servo GPIO %d → center (%d°)", self.pin, self.center_angle)
        self.set_angle(self.center_angle)