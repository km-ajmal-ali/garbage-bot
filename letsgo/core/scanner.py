"""
WasteBot Scanner Module
========================
Handles the SCANNING state – panning the camera through preset
positions and checking for objects at each stop.
"""

import time

from core.config import SCAN_POSITIONS, SCAN_DWELL
from core.states import STATE_APPROACH, STATE_SEARCH_ROTATE
from core.detection import run_detection, pick_best_target
from core.logger import get_logger

log = get_logger("scanner")

# Number of stale frames to discard after moving the servo.
# The CSI camera (Picamera2) buffers frames internally, so the first
# frames after a servo move were actually captured BEFORE the move.
BUFFER_FLUSH_FRAMES = 3

# Number of detection attempts per scan position.
# Taking multiple samples increases reliability.
SAMPLES_PER_POSITION = 3


class Scanner:
    """Controls the camera-pan scanning sweep."""

    def __init__(self, model, camera, pan_servo, display):
        self.model     = model
        self.camera    = camera
        self.pan_servo = pan_servo
        self.display   = display
        self.scan_index = 0
        log.info("Scanner initialised with %d positions: %s",
                 len(SCAN_POSITIONS), SCAN_POSITIONS)

    def reset(self):
        """Restart the sweep from the first position."""
        log.info("Scan sweep reset → starting from position 0")
        self.scan_index = 0

    def _flush_camera_buffer(self):
        """
        Discard stale buffered frames so the next read() returns
        a frame captured AFTER the servo has finished moving.
        """
        log.debug("Flushing %d stale camera buffer frames", BUFFER_FLUSH_FRAMES)
        for _ in range(BUFFER_FLUSH_FRAMES):
            self.camera.read()
            
        log.debug("[DEBUG-SCANNER] _flush_camera_buffer complete")

    def step(self, current_state: str) -> tuple[str, dict | None]:
        """
        Execute one step of the scanning sweep.

        Returns:
            (next_state, target_detection_or_None)
        """
        if self.scan_index >= len(SCAN_POSITIONS):
            log.warning("Full sweep complete – no objects detected across %d positions",
                        len(SCAN_POSITIONS))
            self.reset()
            return STATE_SEARCH_ROTATE, None

        angle = SCAN_POSITIONS[self.scan_index]

        # Move servo and let it settle by pumping camera frames
        log.info("SCAN step %d/%d → pan servo to %d°",
                 self.scan_index + 1, len(SCAN_POSITIONS), angle)
        self.pan_servo.servo.angle = angle
        
        # ACTIVE WAIT: Instead of blocking with time.sleep(), we continuously 
        # ingest camera frames to ensure no underlying driver buffers overload
        # while the physical servo settles into position.
        start_time = time.time()
        flush_count = 0
        while (time.time() - start_time) < SCAN_DWELL:
            self.camera.read()
            flush_count += 1
            time.sleep(0.005)  # Let CPU breathe
            
        log.debug("Servo settled after %.2fs dwell. Pumped %d frames.", SCAN_DWELL, flush_count)

        # Take multiple detection samples at this position
        for attempt in range(SAMPLES_PER_POSITION):
            log.debug("[DEBUG-SCANNER] About to read attempt %d", attempt + 1)
            ret, frame = self.camera.read()
            log.debug("[DEBUG-SCANNER] Read complete: ret=%s", ret)
            
            if not ret or frame is None:
                log.warning("Camera read failed at pan=%d°  attempt=%d", angle, attempt + 1)
                continue

            log.debug("[DEBUG-SCANNER] About to run_detection")
            detections = run_detection(self.model, frame, camera=self.camera)
            log.debug("[DEBUG-SCANNER] run_detection complete")
            self.display.show(frame, detections, current_state)

            log.info("SCAN pan=%d°  sample=%d/%d  detections=%d",
                     angle, attempt + 1, SAMPLES_PER_POSITION, len(detections))

            # Log all detections at this position
            for i, det in enumerate(detections):
                log.debug("  detection[%d]: '%s' conf=%.2f bbox=%s depth≈%.0fcm",
                          i, det['label'], det['confidence'], det['bbox'],
                          det.get('depth_cm', -1))

            if detections:
                target = pick_best_target(detections)
                if target:
                    depth = target.get('depth_cm', -1)
                    log.info("✓ TARGET FOUND: '%s' conf=%.2f at pan=%d°  depth≈%.0fcm  → APPROACH",
                             target['label'], target['confidence'], angle, depth)
                    return STATE_APPROACH, target

        log.debug("No detections at pan=%d° after %d samples → next position",
                  angle, SAMPLES_PER_POSITION)
        self.scan_index += 1
        return current_state, None
