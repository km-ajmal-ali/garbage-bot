import sys
import os

# Ensure letsgo is in the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from common.motors import MotorControl

def main():
    print("=== Motor Testing Tool ===")
    print("Enter the 6 GPIO pins for the motor driver.")
    print("Format: IN1,IN2,IN3,IN4,EN_A,EN_B (e.g., 17,27,22,23,24,25)")
    
    pin_input = input("Pins: ").strip()
    try:
        pins = [int(p.strip()) for p in pin_input.split(',')]
        if len(pins) != 6:
            raise ValueError("Expected exactly 6 pins.")
    except Exception as e:
        print(f"Invalid input: {e}")
        return

    print(f"Initializing MotorControl with pins: {pins}")
    try:
        motor = MotorControl(pins)
    except Exception as e:
        print(f"Failed to initialize motors: {e}")
        return

    print("\nMotor Controller Initialized.")
    print("Commands:")
    print("  F - Move Forward")
    print("  B - Move Backward")
    print("  L - Move Left")
    print("  R - Move Right")
    print("  S - Stop")
    print("  SPEED <value> - Adjust Speed (0-100)")
    print("  Q - Quit / Cleanup")

    speed = 50

    try:
        while True:
            cmd = input("\nEnter command: ").strip().upper()
            if cmd == 'F':
                print(f"Moving forward at speed {speed}...")
                motor.move("forward", speed)
            elif cmd == 'B':
                print(f"Moving backward at speed {speed}...")
                motor.move("backward", speed)
            elif cmd == 'L':
                print(f"Moving left at speed {speed}...")
                motor.move("left", speed)
            elif cmd == 'R':
                print(f"Moving right at speed {speed}...")
                motor.move("right", speed)
            elif cmd == 'S':
                print("Stopping motors...")
                motor.stop()
            elif cmd.startswith('SPEED'):
                try:
                    speed = int(cmd.split()[1])
                    print(f"Speed set to {speed}%")
                except:
                    print("Invalid speed format. Use e.g., 'SPEED 70'")
            elif cmd == 'Q':
                print("Quitting...")
                break
            else:
                print("Unknown command. Valid commands: F, B, L, R, S, SPEED <val>, Q")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up...")
        motor.cleanup()
        print("Done.")

if __name__ == "__main__":
    main()