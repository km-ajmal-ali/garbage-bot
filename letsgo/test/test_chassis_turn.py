"""
WasteBot – Chassis Turn Calibration Tool
==========================================

Interactive tool to find the exact time constant for the chassis
to perform precise turns (e.g. 15°, 30°, 45°, 90°, 180°, 360°).

Usage (run on the Raspberry Pi):
    cd letsgo
    python3 test/test_chassis_turn.py

Workflow:
    1.  Place the robot on the floor and mark the front direction.
    2.  Choose a target angle (e.g. 90°).
    3.  Adjust the turn duration until the robot turns exactly that angle.
    4.  The tool will print the calibrated CHASSIS_DEGREES_PER_SEC value.
    5.  Copy that value into core/config.py.

Uses:
    - common.motors.MotorControl  (motor driver)
    - core.config                 (GPIO pins, speeds)
"""

import sys
import os
import time

# ── Ensure letsgo is on the path ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from core.logger import setup_logging, get_logger
setup_logging(level="INFO")
log = get_logger("turn_cal")

from core.config import (
    MOTOR_PINS,
    CHASSIS_TURN_SPEED,
    CHASSIS_DEGREES_PER_SEC,
)
from common.motors import MotorControl


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def do_turn(motor: MotorControl, direction: str, speed: int, duration: float):
    """Execute a timed turn and return the actual elapsed time."""
    print(f"\n    ▶  Turning {direction.upper()} at {speed}% for {duration:.3f}s …")
    t0 = time.time()
    motor.move(direction, speed)
    time.sleep(duration)
    motor.stop()
    elapsed = time.time() - t0
    print(f"    ■  Done  (actual elapsed: {elapsed:.3f}s)")
    return elapsed


def calc_rate(angle: float, duration: float) -> float:
    """Return degrees/second from a measured turn."""
    if duration <= 0:
        return 0.0
    return angle / duration


def print_header():
    print()
    print("=" * 60)
    print("  WasteBot – Chassis Turn Calibration")
    print("=" * 60)
    print()
    print(f"  Motor pins      : {MOTOR_PINS}")
    print(f"  Turn speed      : {CHASSIS_TURN_SPEED}%")
    print(f"  Current rate    : {CHASSIS_DEGREES_PER_SEC}°/s")
    print()


def print_menu():
    print("─" * 60)
    print("  Commands:")
    print("    1  –  Quick test: turn RIGHT for a custom duration")
    print("    2  –  Quick test: turn LEFT  for a custom duration")
    print("    3  –  Calibrate: find °/s by doing a known-angle turn")
    print("    4  –  Verify: do an exact N° turn using current rate")
    print("    5  –  Full 360° calibration run")
    print("    6  –  Change speed")
    print("    Q  –  Quit")
    print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration routines
# ═══════════════════════════════════════════════════════════════════════════

def cmd_quick_turn(motor, direction, speed):
    """Turn for a user-specified duration."""
    try:
        duration = float(input(f"  Duration (seconds) for {direction.upper()} turn: "))
    except (ValueError, EOFError):
        print("  Invalid input.")
        return

    do_turn(motor, direction, speed, duration)
    print("  → Observe how far the robot turned. Adjust duration and repeat.")


def cmd_calibrate(motor, speed):
    """
    Ask the user for a target angle, let them adjust the duration until
    the robot turns exactly that angle, then calculate °/s.
    """
    try:
        angle = float(input("  Target angle to calibrate (e.g. 90): "))
        if angle <= 0:
            print("  Angle must be positive.")
            return
    except (ValueError, EOFError):
        print("  Invalid input.")
        return

    direction = input("  Direction (L/R) [R]: ").strip().upper()
    direction = "left" if direction == "L" else "right"

    # Start with current rate estimate
    rate = CHASSIS_DEGREES_PER_SEC
    duration = angle / rate if rate > 0 else 1.0

    print(f"\n  Target: {angle}° {direction.upper()}")
    print(f"  Starting estimate: {duration:.3f}s  (based on {rate:.1f}°/s)")
    print()
    print("  Loop: adjust the duration until the turn is exact.")
    print("  Type a new duration, or press ENTER to re-run the same,")
    print("  or type 'done' when the turn is perfect.\n")

    while True:
        elapsed = do_turn(motor, direction, speed, duration)

        cmd = input(f"\n  New duration (or ENTER to repeat, 'done' to accept): ").strip()

        if cmd.lower() == "done":
            rate = calc_rate(angle, duration)
            print()
            print("  ┌──────────────────────────────────────────┐")
            print(f"  │  ✓ Calibration result                    │")
            print(f"  │                                          │")
            print(f"  │  Angle     : {angle:>6.1f}°                    │")
            print(f"  │  Duration  : {duration:>6.3f}s                    │")
            print(f"  │  Speed     : {speed:>4d}%                      │")
            print(f"  │  Rate      : {rate:>6.1f}°/s                   │")
            print(f"  │                                          │")
            print(f"  │  → Set CHASSIS_DEGREES_PER_SEC = {rate:.1f}    │")
            print(f"  │    in core/config.py                     │")
            print("  └──────────────────────────────────────────┘")
            return

        if cmd == "":
            continue  # repeat same duration

        try:
            duration = float(cmd)
        except ValueError:
            print("  Invalid number. Try again.")


def cmd_verify(motor, speed):
    """
    Use the current CHASSIS_DEGREES_PER_SEC to perform an exact N° turn.
    Lets the user see if the calibration is correct.
    """
    rate = CHASSIS_DEGREES_PER_SEC

    try:
        angle = float(input(f"  Turn angle (degrees) [90]: ").strip() or "90")
    except (ValueError, EOFError):
        print("  Invalid input.")
        return

    direction = input("  Direction (L/R) [R]: ").strip().upper()
    direction = "left" if direction == "L" else "right"

    duration = angle / rate
    print(f"\n  Calculated: {angle}° at {rate}°/s → {duration:.3f}s")

    do_turn(motor, direction, speed, duration)
    print(f"\n  Did the robot turn exactly {angle}°?")
    print(f"  If not, use command 3 (Calibrate) to find the right rate.")


def cmd_full_360(motor, speed):
    """
    Do a full 360° turn and let the user time it, then calculate °/s.
    """
    direction = input("  Direction (L/R) [R]: ").strip().upper()
    direction = "left" if direction == "L" else "right"

    print(f"\n  The robot will turn {direction.upper()} continuously.")
    print("  Press ENTER to start, then press ENTER again when it")
    print("  completes exactly one full rotation (360°).")
    input("\n  Press ENTER to START … ")

    t0 = time.time()
    motor.move(direction, speed)

    input("  Press ENTER to STOP (when 360° is complete) … ")

    motor.stop()
    elapsed = time.time() - t0

    rate = calc_rate(360.0, elapsed)
    print(f"\n  Elapsed: {elapsed:.2f}s for 360°")
    print(f"  Rate   : {rate:.1f}°/s")
    print()
    print(f"  → Set CHASSIS_DEGREES_PER_SEC = {rate:.1f} in core/config.py")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print_header()

    print("Initialising motors …")
    try:
        motor = MotorControl(MOTOR_PINS)
    except Exception as e:
        print(f"Failed to initialise motors: {e}")
        return

    speed = CHASSIS_TURN_SPEED
    print(f"Motors ready.  Turn speed = {speed}%\n")

    try:
        while True:
            print_menu()
            cmd = input("\n  Enter command: ").strip().upper()

            if cmd == "1":
                cmd_quick_turn(motor, "right", speed)
            elif cmd == "2":
                cmd_quick_turn(motor, "left", speed)
            elif cmd == "3":
                cmd_calibrate(motor, speed)
            elif cmd == "4":
                cmd_verify(motor, speed)
            elif cmd == "5":
                cmd_full_360(motor, speed)
            elif cmd == "6":
                try:
                    speed = int(input(f"  New speed (0–100) [{speed}]: ").strip() or str(speed))
                    print(f"  Speed set to {speed}%")
                except ValueError:
                    print("  Invalid number.")
            elif cmd == "Q":
                print("\nQuitting …")
                break
            else:
                print("  Unknown command.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("Cleaning up motors …")
        motor.stop()
        motor.cleanup()
        print("Done. ✓")


if __name__ == "__main__":
    main()
