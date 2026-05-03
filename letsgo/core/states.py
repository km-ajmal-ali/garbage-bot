"""
WasteBot State Definitions
===========================
Finite-state machine states for the robot's behaviour loop.
"""

STATE_SCANNING      = "SCANNING"
STATE_APPROACH      = "APPROACH"
STATE_TILT_ADJUST   = "TILT_ADJUST"
STATE_COLLECT       = "COLLECT"
STATE_SEARCH_ROTATE = "SEARCH_ROTATE"

# QR Code Destination States
STATE_SEARCH_QR        = "SEARCH_QR"
STATE_SEARCH_ROTATE_QR = "SEARCH_ROTATE_QR"
STATE_APPROACH_QR      = "APPROACH_QR"
STATE_DROP_QR          = "DROP_QR"
