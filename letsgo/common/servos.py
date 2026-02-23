from gpiozero import AngularServo
from time import sleep

# Attempt to use pigpio factory for smoother servo movement
try:
    from gpiozero.pins.pigpio import PiGPIOFactory
    from gpiozero import Device
    # Only set if not already set (avoid re-initialization)
    if Device.pin_factory is None or not isinstance(Device.pin_factory, PiGPIOFactory):
        print("Initializing pigpio factory for smooth servo control...")
        Device.pin_factory = PiGPIOFactory()
except ImportError:
    print("Warning: pigpio python library not found. Install with 'sudo apt install python3-pigpio'")
except Exception as e:
    print(f"Warning: Could not connect to pigpio daemon: {e}. Servos may jitter.")
    print("Ensure pigpiod is running: 'sudo pigpiod'")
    # Default factory will be used automatically.

class CameraServo:
    def __init__(self, pin=25, min_angle=-90, max_angle=90):
        """
        Initializes a pan servo for the camera.
        Default pin is GPIO 25.
        Adjust min_pulse_width and max_pulse_width based on your servo datasheet
        (Common values are 0.0005 to 0.0025).
        """
        # Ensure we are using the best factory available
        self.servo = AngularServo(pin, 
                                  min_angle=min_angle, 
                                  max_angle=max_angle,
                                  min_pulse_width=0.0005, 
                                  max_pulse_width=0.0025)
        self.current_angle = 0
        self.servo.angle = self.current_angle

    def set_angle(self, angle):
        """Sets the servo to a specific angle."""
        if -90 <= angle <= 90:
            self.servo.angle = angle
            self.current_angle = angle
            sleep(0.3)  # Give time for physical movement
        else:
            print("Angle out of range (-90 to 90)")

    def look_around(self):
        """
        A routine for the robot to scan its environment.
        Moves the camera in steps to find garbage.
        """
        print("Scanning environment...")
        positions = [0, 45, 90, 45, 0, -45, -90, -45, 0]
        for pos in positions:
            self.set_angle(pos)
            # In your main.py, you would call the detector here
            sleep(0.5)

    def center(self):
        """Resets camera to front-facing."""
        self.set_angle(0)