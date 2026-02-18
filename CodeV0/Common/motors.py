import RPi.GPIO as GPIO

class MotorControl:
    def __init__(self, pins):
        self.pins = pins # [IN1, IN2, IN3, IN4, EN_A, EN_B]
        GPIO.setmode(GPIO.BCM)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
        self.pwm_a = GPIO.PWM(pins[4], 1000)
        self.pwm_b = GPIO.PWM(pins[5], 1000)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    def move(self, direction, speed=50):
        if direction == "forward":
            GPIO.output(self.pins[0], True); GPIO.output(self.pins[1], False)
            GPIO.output(self.pins[2], True); GPIO.output(self.pins[3], False)
        elif direction == "left":
            GPIO.output(self.pins[0], False); GPIO.output(self.pins[1], True)
            GPIO.output(self.pins[2], True); GPIO.output(self.pins[3], False)
        elif direction == "right":
            GPIO.output(self.pins[0], True); GPIO.output(self.pins[1], False)
            GPIO.output(self.pins[2], False); GPIO.output(self.pins[3], True)
        elif direction == "backward":
            GPIO.output(self.pins[0], False); GPIO.output(self.pins[1], True)
            GPIO.output(self.pins[2], False); GPIO.output(self.pins[3], True)
        
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)

    def stop(self):
        GPIO.output(self.pins[0], False); GPIO.output(self.pins[1], False)
        GPIO.output(self.pins[2], False); GPIO.output(self.pins[3], False)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        
    def cleanup(self):
        self.stop()
        GPIO.cleanup()