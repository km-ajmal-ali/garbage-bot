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
    def __init__(self, pin=25, min_angle=-90, max_angle=90):
        """
        Initializes a servo for camera control.
        Adjust min_pulse_width and max_pulse_width based on your servo datasheet
        (Common values are 0.0005 to 0.0025).
        """
        log.info("Init servo on GPIO %d  range=[%d°, %d°]", pin, min_angle, max_angle)
        self.pin = pin
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = 0
        log.info("Servo GPIO %d initialised (PWM off)", pin)

    def set_angle(self, angle):
        """Sets the servo to a specific angle (includes 0.3s settle sleep, then disables PWM to prevent jitter)."""
        if -90 <= angle <= 90:
            self.current_angle = angle
            log.debug("Servo GPIO %d → %d°  (settling 0.3s)", self.pin, angle)
            
            # Instantiate the servo specifically to move it, then close it entirely.
            # This completely releases the GPIO pin, perfectly mimicking the behavior
            # of the Python script exiting, shutting down the PWM completely.
            servo = AngularServo(self.pin,
                                 min_angle=self.min_angle,
                                 max_angle=self.max_angle,
                                 min_pulse_width=0.0005,
                                 max_pulse_width=0.0025,
                                 initial_angle=angle)
            sleep(0.3)  # Give time for physical movement
            servo.close() # Free the pin and kill the signal
        else:
            log.warning("Servo GPIO %d: angle %d° out of range [-90, 90]", self.pin, angle)

    def look_around(self):
        """
        A routine for the robot to scan its environment.
        Moves the camera in steps to find garbage.
        """
        log.info("Servo GPIO %d: look_around sweep starting", self.pin)
        positions = [0, 45, 90, 45, 0, -45, -90, -45, 0]
        for pos in positions:
            self.set_angle(pos)
            sleep(0.5)
        log.info("Servo GPIO %d: look_around sweep complete", self.pin)

    def center(self):
        """Resets camera to front-facing."""
        log.debug("Servo GPIO %d → center (0°)", self.pin)
        self.set_angle(0)