from gpiozero import DigitalOutputDevice, PWMOutputDevice
from core.logger import get_logger

log = get_logger("motor")


class MotorControl:
    """
    L298N dual-motor driver using gpiozero (Pi 5 compatible).

    Pin list: [IN1, IN2, IN3, IN4, EN_A, EN_B]
      • IN1/IN2 control Motor A direction
      • IN3/IN4 control Motor B direction
      • EN_A / EN_B are PWM speed pins
    """

    def __init__(self, pins):
        self.pins = pins  # [IN1, IN2, IN3, IN4, EN_A, EN_B]

        log.info(f"Initialising motor driver on pins {pins}")

        # Direction pins (digital HIGH / LOW)
        self.in1 = DigitalOutputDevice(pins[0])
        self.in2 = DigitalOutputDevice(pins[1])
        self.in3 = DigitalOutputDevice(pins[2])
        self.in4 = DigitalOutputDevice(pins[3])

        # Speed pins (PWM, 1 kHz)
        self.pwm_a = PWMOutputDevice(pins[4], frequency=1000)
        self.pwm_b = PWMOutputDevice(pins[5], frequency=1000)

        log.info("Motor driver ready (IN1=%d IN2=%d IN3=%d IN4=%d EN_A=%d EN_B=%d)",
                 *pins)

    def _set_direction(self, in1, in2, in3, in4):
        """Set the four direction pins (True/False)."""
        self.in1.value = in1
        self.in2.value = in2
        self.in3.value = in3
        self.in4.value = in4

    def move(self, direction, speed=50):
        """
        Move the chassis in the given direction.

        Args:
            direction: 'forward', 'backward', 'left', or 'right'.
            speed:     0-100 (percentage of full power).
        """
        if direction == "forward":
            self._set_direction(True, False, True, False)
        elif direction == "left":
            self._set_direction(False, True, True, False)
        elif direction == "right":
            self._set_direction(True, False, False, True)
        elif direction == "backward":
            self._set_direction(False, True, False, True)
        else:
            log.warning("Unknown direction '%s' – ignoring", direction)
            return

        # gpiozero PWM uses 0.0–1.0 range
        duty = max(0.0, min(speed / 100.0, 1.0))
        self.pwm_a.value = duty
        self.pwm_b.value = duty

        log.info("MOVE  %-8s  speed=%d%%  duty=%.2f", direction, speed, duty)

    def stop(self):
        """Stop both motors immediately."""
        self._set_direction(False, False, False, False)
        self.pwm_a.value = 0
        self.pwm_b.value = 0
        log.debug("STOP  (all pins LOW, PWM=0)")

    def cleanup(self):
        """Release all GPIO resources."""
        log.info("Cleaning up motor GPIO resources")
        self.stop()
        self.in1.close()
        self.in2.close()
        self.in3.close()
        self.in4.close()
        self.pwm_a.close()
        self.pwm_b.close()
        log.info("Motor GPIO released")