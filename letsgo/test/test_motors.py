import sys
import os
from time import sleep

# Ensure project root is in the path so we can import CodeV0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PARENT_DIR)
sys.path.insert(0, PROJECT_DIR)

from CodeV0.Common.motors import MotorControl

def main():
    print("=== Motor Tester (CodeV0 Library) ===")
    
    # Default pins used in CodeV0 main.py
    default_pins = "17, 18, 22, 23, 12, 13"
    print("Enter GPIO pins for the motors (IN1, IN2, IN3, IN4, EN_A, EN_B)")
    pin_str = input(f"Enter 6 pins separated by commas [default: {default_pins}]: ").strip()
    
    if not pin_str:
        pin_str = default_pins
        
    try:
        pins = [int(p.strip()) for p in pin_str.split(',')]
        if len(pins) != 6:
            print(f"Error: Expected 6 pins, but got {len(pins)}.")
            return
    except ValueError:
        print("Invalid pin format. Please enter numbers separated by commas.")
        return

    print(f"Initializing motors on pins: {pins}...")
    try:
        motors = MotorControl(pins)
    except Exception as e:
        print(f"Failed to initialize motors: {e}")
        return
        
    print("Motors ready.")
    print("Commands:")
    print("  w: forward")
    print("  s: backward")
    print("  a: left")
    print("  d: right")
    print("  x: stop")
    print("  speed <value>: set speed (0-100), e.g., 'speed 75'")
    print("  q: quit")
    
    current_speed = 50.0
    
    try:
        while True:
            cmd_input = input(f"Enter command (w/a/s/d/x | speed <val> | q) [Speed: {current_speed}]: ").lower().strip()
            
            if not cmd_input:
                continue
                
            if cmd_input == 'q':
                break
                
            if cmd_input.startswith('speed'):
                parts = cmd_input.split()
                if len(parts) > 1:
                    try:
                        val = float(parts[1])
                        if 0 <= val <= 100:
                            current_speed = val
                            print(f"Speed set to {current_speed}")
                        else:
                            print("Speed out of range. Please enter a value between 0 and 100.")
                    except ValueError:
                        print("Invalid speed value.")
                else:
                    print("Usage: speed <value>")
                continue
                
            if cmd_input == 'w':
                print(f"Moving forward at speed {current_speed}...")
                motors.move('forward', current_speed)
            elif cmd_input == 's':
                print(f"Moving backward at speed {current_speed}...")
                motors.move('backward', current_speed)
            elif cmd_input == 'a':
                print(f"Moving left at speed {current_speed}...")
                motors.move('left', current_speed)
            elif cmd_input == 'd':
                print(f"Moving right at speed {current_speed}...")
                motors.move('right', current_speed)
            elif cmd_input == 'x':
                print("Stopping motors...")
                motors.stop()
            else:
                print("Invalid command. Use w, a, s, d, x, 'speed <val>', or q.")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up and exiting...")
        motors.cleanup()

if __name__ == "__main__":
    main()
