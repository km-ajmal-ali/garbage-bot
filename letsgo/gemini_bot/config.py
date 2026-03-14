"""
GeminiBot Configuration
========================
All tunable constants for the Gemini-vision powered garbage collector.
Edit this file to calibrate for your exact hardware setup.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════
# GOOGLE GEMINI API
# ═══════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY = "AIzaSyD_RANDOM_PLACEHOLDER_REPLACE_ME_1234"
GEMINI_MODEL   = "gemini-2.0-flash"   # fast + cheap, good vision

# ═══════════════════════════════════════════════════════════════════════════
# GPIO PINS  (same as letsgo)
# ═══════════════════════════════════════════════════════════════════════════

# Motor driver (L298N)  [IN1, IN2, IN3, IN4, EN_A, EN_B]
MOTOR_PINS = [17, 27, 22, 23, 24, 25]

# Servo GPIO pins
PAN_SERVO_PIN  = 13   # X-axis (horizontal sweep)
TILT_SERVO_PIN = 12   # Y-axis (vertical / look-up-down)

# ═══════════════════════════════════════════════════════════════════════════
# MOTOR SPEEDS  (PWM duty cycle 0-100)
# ═══════════════════════════════════════════════════════════════════════════
APPROACH_SPEED      = 60   # driving toward an object
COLLECT_SPEED       = 55   # slow crawl towards the object
SEARCH_ROTATE_SPEED = 65   # in-place rotation during search

# ═══════════════════════════════════════════════════════════════════════════
# PAN SERVO LIMITS  (X-axis / horizontal sweep)
# ═══════════════════════════════════════════════════════════════════════════
PAN_MIN_ANGLE    = -90   # maximum left pan angle (degrees)
PAN_MAX_ANGLE    =  90   # maximum right pan angle (degrees)
PAN_CENTER_ANGLE =   0   # neutral / forward-facing angle
PAN_SCAN_STEP    =  30   # degrees per scan sweep increment

# Auto-generate scan positions
SCAN_POSITIONS = list(range(PAN_MIN_ANGLE, PAN_MAX_ANGLE + 1, PAN_SCAN_STEP))

# ═══════════════════════════════════════════════════════════════════════════
# TILT SERVO LIMITS  (Y-axis / vertical look up-down)
# ═══════════════════════════════════════════════════════════════════════════
TILT_MIN_ANGLE    = -30
TILT_MAX_ANGLE    =  60
TILT_CENTER_ANGLE = -20
TILT_STEP         =  10
TILT_SETTLE       =  0.10

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════════════
CAM_WIDTH         = 800
CAM_HEIGHT        = 800
CAMERA_ROTATE_180 = True   # True if the camera is mounted upside-down

# ═══════════════════════════════════════════════════════════════════════════
# SERVO MECHANICAL STABILISE
# ═══════════════════════════════════════════════════════════════════════════
SERVO_STABILIZE_DELAY = 1.0   # seconds after servo stops for vibration to die

# ═══════════════════════════════════════════════════════════════════════════
# SCANNING PHASE  (Gemini-specific)
# ═══════════════════════════════════════════════════════════════════════════
SCAN_DWELL = 0.25                # seconds to wait after servo move (settle)
SCAN_IMAGE_INTERVAL = 0.5        # seconds between captures at the same angle
SCAN_SAMPLES_PER_POSITION = 1    # only 1 image per position to reduce API cost

# ═══════════════════════════════════════════════════════════════════════════
# COLLECTION / APPROACH PHASE  (Gemini-specific)
# ═══════════════════════════════════════════════════════════════════════════
APPROACH_IMAGE_INTERVAL = 2.0    # seconds between Gemini calls during approach
APPROACH_DRIVE_PULSE    = 0.5    # seconds of forward drive per pulse
APPROACH_STEER_PULSE    = 0.15   # seconds of steering correction per pulse
MAX_APPROACH_CYCLES     = 30     # max Gemini calls before giving up approach
MAX_LOST_ATTEMPTS       = 3      # times Gemini says "no object" before aborting

# ═══════════════════════════════════════════════════════════════════════════
# SEARCH ROTATE  (when full sweep finds nothing)
# ═══════════════════════════════════════════════════════════════════════════
ROTATE_STEP_TIME  = 0.8
MAX_SEARCH_STEPS  = 12

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY  (JPEG compression for API payload)
# ═══════════════════════════════════════════════════════════════════════════
JPEG_QUALITY    = 60   # 0-100 — lower = smaller payload = cheaper
RESIZE_WIDTH    = 640  # resize image before sending (saves tokens)
RESIZE_HEIGHT   = 480
