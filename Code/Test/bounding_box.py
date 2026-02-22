"""
Bounding Box & Depth Estimation Test Script
=============================================
Uses YOLOv8m model for real-time object detection.
Calculates bounding boxes, estimated depth, and displays
all metrics in a live window.

Functions are written modularly so they can be imported
and reused after testing.

Dependencies:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Models", "yolov8m.pt")
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 640  # YOLO input resolution

# Focal length (pixels) and known average object width (cm) for depth estimation.
# These are rough defaults – calibrate with your actual camera for accuracy.
FOCAL_LENGTH_PX = 600       # approximate focal length in pixels (typical webcam)
KNOWN_OBJECT_WIDTH_CM = 20  # assumed average real-world width of detected objects (cm)

# Colour palette for different classes (BGR)
COLORS = [
    (0, 255, 127),   # spring green
    (255, 100, 50),   # coral blue
    (50, 200, 255),   # amber
    (200, 50, 255),   # magenta
    (100, 255, 100),  # light green
    (255, 255, 50),   # cyan
    (50, 50, 255),    # red
    (255, 50, 200),   # pink
]


# ===================================================================
# Core Functions (reusable)
# ===================================================================

def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """
    Load a YOLOv8 model from the given path.
    If the .pt file doesn't exist locally it will be downloaded
    automatically by ultralytics (e.g. 'yolov8m.pt').

    Args:
        model_path: Path to the YOLOv8 weights file (.pt).

    Returns:
        A YOLO model instance ready for inference.
    """
    if not os.path.isfile(model_path):
        print(f"[INFO] Model not found at '{model_path}'. "
              f"Falling back to auto-download of 'yolov8m.pt'...")
        model_path = "yolov8m.pt"  # ultralytics will download it

    model = YOLO(model_path)
    print(f"[INFO] Model loaded: {model_path}")
    return model


def detect_objects(model: YOLO, frame: np.ndarray,
                   conf_threshold: float = CONFIDENCE_THRESHOLD):
    """
    Run YOLOv8 inference on a single frame and return structured
    detection results.

    Args:
        model:          YOLO model instance.
        frame:          BGR image (numpy array).
        conf_threshold: Minimum confidence to keep a detection.

    Returns:
        List of dicts, each containing:
            - 'label'      : class name (str)
            - 'class_id'   : class index (int)
            - 'confidence' : detection confidence (float)
            - 'bbox'       : (x1, y1, x2, y2) in pixel coords (ints)
            - 'center'     : (cx, cy) center of bounding box (ints)
            - 'width_px'   : bounding box width in pixels (int)
            - 'height_px'  : bounding box height in pixels (int)
    """
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        imgsz=INPUT_SIZE,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = model.names[cls_id]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1

            detections.append({
                'label': label,
                'class_id': cls_id,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'width_px': w,
                'height_px': h,
            })

    return detections


def estimate_depth(bbox_width_px: int,
                   known_width_cm: float = KNOWN_OBJECT_WIDTH_CM,
                   focal_length_px: float = FOCAL_LENGTH_PX) -> float:
    """
    Estimate approximate distance (depth) of an object from the camera
    using the pinhole camera model:
        depth = (known_width * focal_length) / bbox_width_px

    Args:
        bbox_width_px:   Width of the bounding box in pixels.
        known_width_cm:  Known (or assumed) real-world width in cm.
        focal_length_px: Focal length of camera in pixels.

    Returns:
        Estimated depth in centimetres. Returns -1 if calculation is
        not possible (zero-width box).
    """
    if bbox_width_px <= 0:
        return -1.0
    depth_cm = (known_width_cm * focal_length_px) / bbox_width_px
    return round(depth_cm, 1)


def draw_bounding_boxes(frame: np.ndarray,
                        detections: list,
                        show_depth: bool = True) -> np.ndarray:
    """
    Draw bounding boxes, labels, confidence scores, and optional
    depth estimates on the frame.

    Args:
        frame:      BGR image (numpy array) to draw on (modified in-place).
        detections: List of detection dicts from `detect_objects()`.
        show_depth: If True, compute and display depth estimate.

    Returns:
        The annotated frame (same reference as input).
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['confidence']
        cls_id = det['class_id']
        w_px = det['width_px']

        # Pick a consistent colour per class
        color = COLORS[cls_id % len(COLORS)]

        # --- Bounding box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # --- Label + confidence ---
        text = f"{label} {conf:.0%}"

        # --- Depth ---
        if show_depth:
            depth_cm = estimate_depth(w_px)
            det['depth_cm'] = depth_cm  # attach to dict for later use
            if depth_cm > 0:
                text += f" | ~{depth_cm:.0f}cm"

        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Center dot ---
        cx, cy = det['center']
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_hud(frame: np.ndarray, fps: float, num_detections: int) -> np.ndarray:
    """
    Draw a heads-up display overlay with FPS and detection count.

    Args:
        frame:          BGR image.
        fps:            Current frames-per-second.
        num_detections: Number of objects detected in this frame.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info = f"FPS: {fps:.1f}  |  Objects: {num_detections}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2, cv2.LINE_AA)

    return frame


# ===================================================================
# Camera Initialisation (IMX219 CSI / USB webcam)
# ===================================================================

def _try_picamera2(width: int = 640, height: int = 480):
    """
    Attempt to open the camera via Picamera2 (native libcamera).
    Returns a wrapper object with .read() / .release() / .isOpened()
    matching the OpenCV VideoCapture interface.
    """
    try:
        from picamera2 import Picamera2

        class Picamera2Capture:
            def __init__(self, w, h):
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (w, h), "format": "BGR888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                # Warm-up frame
                self.picam2.capture_array()
                print(f"[INFO] Picamera2 opened ({w}x{h}).")

            def read(self):
                try:
                    frame = self.picam2.capture_array()
                    return True, frame
                except Exception as e:
                    print(f"[WARN] Picamera2 read error: {e}")
                    return False, None

            def release(self):
                self.picam2.stop()
                self.picam2.close()

            def isOpened(self):
                return True

        return Picamera2Capture(width, height)

    except ImportError:
        print("[INFO] Picamera2 not installed – skipping.")
    except Exception as e:
        print(f"[WARN] Picamera2 init failed: {e}")
    return None


def _try_gstreamer_libcamera(width: int = 640, height: int = 480):
    """
    Attempt to open via a GStreamer pipeline using libcamerasrc.
    Works on Raspberry Pi OS when GStreamer is built into OpenCV.
    """
    pipeline = (
        f"libcamerasrc ! "
        f"video/x-raw,width={width},height={height},framerate=30/1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink"
    )
    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] GStreamer libcamera pipeline opened ({width}x{height}).")
                return cap
            cap.release()
    except Exception as e:
        print(f"[WARN] GStreamer pipeline failed: {e}")
    return None


def _try_opencv_v4l2(camera_index: int = 0, width: int = 640, height: int = 480):
    """
    Attempt to open camera via OpenCV V4L2 / generic backend.
    This is the standard fallback for USB webcams.
    """
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_ANY,  "ANY"),
    ]
    for backend, name in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, _ = cap.read()
        if ret:
            print(f"[INFO] OpenCV {name} backend opened index {camera_index} ({width}x{height}).")
            return cap
        cap.release()
    return None


def open_camera(camera_index: int = 0, width: int = 640, height: int = 480):
    """
    Try every available camera backend in priority order and return
    the first one that works.

    Priority:
        1. Picamera2  (native CSI – IMX219, etc.)
        2. GStreamer libcamera pipeline
        3. OpenCV V4L2 / generic (USB webcams)

    Returns:
        An object with .read(), .release(), .isOpened() methods,
        or None if nothing works.
    """
    print("[INFO] Probing camera backends...")

    cap = _try_picamera2(width, height)
    if cap:
        return cap

    cap = _try_gstreamer_libcamera(width, height)
    if cap:
        return cap

    cap = _try_opencv_v4l2(camera_index, width, height)
    if cap:
        return cap

    return None


# ===================================================================
# Real-time Test Loop
# ===================================================================

def run_realtime_test(camera_index: int = 0):
    """
    Open a camera feed (CSI or USB), run YOLOv8m detection, and
    display the annotated output in a window. Press 'q' to quit.

    Args:
        camera_index: Index of the camera device (used for USB fallback).
    """
    # Load model
    model = load_model()

    # Open camera (auto-detect best backend)
    cap = open_camera(camera_index, width=640, height=480)
    if cap is None:
        print("[ERROR] Could not open camera on any backend.")
        print("        Ensure Picamera2 is installed:  pip install picamera2")
        return

    print("[INFO] Camera ready. Press 'q' to quit.")
    print("=" * 50)

    prev_time = time.time()
    fps = 0.0

    window_name = "YOLOv8m - Bounding Box & Depth Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Retrying...")
            time.sleep(0.1)  # small delay to avoid busy-loop
            continue

        # --- Detection ---
        detections = detect_objects(model, frame)

        # --- Draw bounding boxes + depth ---
        draw_bounding_boxes(frame, detections, show_depth=True)

        # --- FPS calculation ---
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # --- HUD overlay ---
        draw_hud(frame, fps, len(detections))

        # --- Print detections to console (for debugging) ---
        if detections:
            for d in detections:
                depth_str = f"{d.get('depth_cm', '?')}cm" if d.get('depth_cm', -1) > 0 else "N/A"
                print(f"  [{d['label']}] conf={d['confidence']:.2f}  "
                      f"bbox={d['bbox']}  depth≈{depth_str}")

        # --- Show ---
        cv2.imshow(window_name, frame)

        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test finished. Window closed.")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    run_realtime_test()
