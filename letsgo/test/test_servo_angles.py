import sys
import os
from time import sleep

# Ensure letsgo is in the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device

# Attempt to use pigpio factory for smoother servo movement
try:
    if Device.pin_factory is None or not isinstance(Device.pin_factory, PiGPIOFactory):
        Device.pin_factory = PiGPIOFactory()
        print("pigpio factory active (smooth movement)")
except Exception as e:
    print(f"Warning: Could not connect to pigpio daemon: {e}")

def main():
    print("=== Servo Angle Tester ===")
    pin_str = input("Enter GPIO pin for the servo (e.g., 25 for Pan): ")
    try:
        pin = int(pin_str)
    except ValueError:
        print("Invalid pin.")
        return

    # Initialize servo with initial_angle=None so it doesn't jump to 0
    print(f"Initializing servo on GPIO {pin} with initial_angle=None...")
    servo = AngularServo(pin,
                         min_angle=-90,
                         max_angle=90,
                         min_pulse_width=0.0005,
                         max_pulse_width=0.0025,
                         initial_angle=None)
    
    print("Servo initialized. It will not move until you enter an angle.")
    
    while True:
        angle_str = input("Enter angle (-90 to 90) or 'q' to quit: ")
        if angle_str.lower() == 'q':
            print("Turning off servo PWM...")
            servo.detach()
            break
            
        try:
            angle = float(angle_str)
            if not (-90 <= angle <= 90):
                print("Angle out of range. Please enter a value between -90 and 90.")
                continue
                
            servo.angle = angle
            print(f"Servo set to {angle}°")
            sleep(0.3)
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    print("Exiting...")

if __name__ == "__main__":
    main()
