"""
WasteBot Configuration
======================
All tunable constants for the robot live here.
Edit this file to calibrate for your exact hardware setup.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(PROJECT_DIR, "models", "yolov8m.hef")

# ═══════════════════════════════════════════════════════════════════════════
# GPIO PINS
# ═══════════════════════════════════════════════════════════════════════════

# Motor driver (L298N)  [IN1, IN2, IN3, IN4, EN_A, EN_B]
MOTOR_PINS = [17, 27, 22, 23, 24, 25]

# Servo GPIO pins
PAN_SERVO_PIN  = 13   # X-axis (horizontal sweep)
TILT_SERVO_PIN = 12   # Y-axis (vertical / look-up-down)

# Collector Servo GPIO Pins.
COLLECTOR_SERVO_PIN = 16
MAX_OPEN_ANGLE = 90
MAX_CLOSE_ANGLE = -10

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR SPEEDS  (PWM duty cycle 0-100)
# ═══════════════════════════════════════════════════════════════════════════
APPROACH_SPEED      = 72   # driving toward an object
COLLECT_SPEED       = 72   # slow crawl over the object
SEARCH_ROTATE_SPEED = 72   # in-place rotation during search

# ═══════════════════════════════════════════════════════════════════════════
# SCANNING
# ═══════════════════════════════════════════════════════════════════════════
SCAN_DWELL     = 0.25   # seconds to wait after servo move (physical settle time)

# ═══════════════════════════════════════════════════════════════════════════
# DEPTH THRESHOLDS  (centimetres)
# ═══════════════════════════════════════════════════════════════════════════
DEPTH_APPROACH_STOP = 25   # stop approaching at this distance
DEPTH_COLLECT_START = 20   # begin collect drive-over
DEPTH_COLLECT_DONE  = 8    # object is under the chassis

# ═══════════════════════════════════════════════════════════════════════════
# ALIGNMENT TOLERANCES  (fraction of frame dimension, 0–1)
# ═══════════════════════════════════════════════════════════════════════════
CENTER_TOLERANCE_X = 0.12   # horizontal dead zone before steering
CENTER_TOLERANCE_Y = 0.15   # vertical dead zone before tilt correction

# ═══════════════════════════════════════════════════════════════════════════
# TIMING
# ═══════════════════════════════════════════════════════════════════════════
ROTATE_STEP_TIME = 0.8     # seconds per search rotation step
MAX_SEARCH_STEPS = 12      # rotations for a full circle
STEER_PULSE_TIME = 0.05    # seconds for a steering correction pulse
DRIVE_PULSE_TIME = 0.02    # seconds for a forward drive pulse

# ═══════════════════════════════════════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.5
MAX_LOST_FRAMES      = 10   # lost frames before giving up during approach
MAX_LOST_COLLECT     = 15   # lost frames before considering object collected

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════════════
CAM_WIDTH        = 800
CAM_HEIGHT       = 800
CAMERA_ROTATE_180 = True   # True if the camera is mounted upside-down

# ═══════════════════════════════════════════════════════════════════════════
# PAN SERVO LIMITS  (X-axis / horizontal sweep)
# ═══════════════════════════════════════════════════════════════════════════
PAN_MIN_ANGLE    = -90   # maximum left pan angle (degrees)
PAN_MAX_ANGLE    =  90   # maximum right pan angle (degrees)
PAN_CENTER_ANGLE =   0   # neutral / forward-facing angle (degrees)
PAN_SCAN_STEP    =  30   # degrees per scan sweep increment

# Auto-generate scan positions from the pan limits
# Sweep from PAN_MIN_ANGLE to PAN_MAX_ANGLE in PAN_SCAN_STEP° steps
SCAN_POSITIONS = list(range(PAN_MIN_ANGLE, PAN_MAX_ANGLE + 1, PAN_SCAN_STEP))

# When the camera is mounted upside-down (CAMERA_ROTATE_180), the pan servo's
# physical direction is inverted relative to the corrected image.  Reverse the
# scan order so the sweep matches the visual left-to-right direction.
if CAMERA_ROTATE_180:
    SCAN_POSITIONS = SCAN_POSITIONS[::-1]

# ═══════════════════════════════════════════════════════════════════════════
# TILT SERVO LIMITS  (Y-axis / vertical look up-down)
# ═══════════════════════════════════════════════════════════════════════════
TILT_MIN_ANGLE    = -20   # maximum downward tilt angle (degrees)
TILT_MAX_ANGLE    =  30  # maximum upward tilt angle (degrees)
TILT_CENTER_ANGLE = -10   # neutral / level angle (degrees)
TILT_STEP         = 10   # tilt increment per adjustment step
TILT_SETTLE       = 0.10  # seconds to wait after tilt move

# ═══════════════════════════════════════════════════════════════════════════
# CHASSIS ALIGNMENT  (rotate body to face the scanned target)
# ═══════════════════════════════════════════════════════════════════════════
CHASSIS_TURN_SPEED     = SEARCH_ROTATE_SPEED  # PWM% used for alignment turns
CHASSIS_DEGREES_PER_SEC = 105.9  # approx turn rate at CHASSIS_TURN_SPEED
                                # calibrate: measure time for 360° turn, then 360/time
PAN_ALIGN_THRESHOLD    = 15     # ignore pan offsets smaller than this (degrees)
# ═══════════════════════════════════════════════════════════════════════════
# SERVO MECHANICAL STABILISE
# ═══════════════════════════════════════════════════════════════════════════
SERVO_STABILIZE_DELAY = 1.0   # seconds to wait after servo stops for mount vibration to settle
