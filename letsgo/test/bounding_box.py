"""
Bounding Box & Depth Estimation Test Script (Hailo AI HAT+ Accelerated)
========================================================================
Uses YOLOv8m HEF model on the Hailo-8L AI HAT+ (26 TOPS) for
real-time object detection at high FPS.

Calculates bounding boxes, estimated depth, and displays all metrics
in a live window.

Functions are written modularly so they can be imported and reused
later after testing.

Dependencies:
    - hailort  (system package: sudo apt install hailort)
    - hailo_platform  (Python bindings, installed with hailort)
    - opencv-python
    - numpy
    - picamera2  (for IMX219 CSI camera)
"""

import cv2
import numpy as np
import os
import time
import threading

import sys
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from core.logger import get_logger
log = get_logger("ha_test")

from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "yolov10s.hef")
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 640  # YOLOv8 input resolution (640x640)

# COCO class names (80 classes) – YOLOv8m default
# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
#     "truck", "boat", "traffic light", "fire hydrant", "stop sign",
#     "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
#     "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
#     "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
#     "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
#     "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
#     "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#     "couch", "potted plant", "bed", "dining table", "toilet", "tv",
#     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
#     "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
#     "scissors", "teddy bear", "hair drier", "toothbrush",
# ]

COCO_CLASSES = [
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object", "object", "object", "object", "object",
    "object", "object", "object",
]


# Focal length (pixels) and known average object width (cm) for depth estimation.
# These are rough defaults – calibrate with your actual camera for accuracy.
FOCAL_LENGTH_PX = 600       # approximate focal length in pixels (IMX219)
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
# Hailo Model Loader (reusable)
# ===================================================================

class HailoYOLOv8:
    """
    Wrapper around the Hailo runtime to load a YOLOv8 HEF model
    and run inference on the AI HAT+ accelerator.

    The HEF is expected to include on-chip NMS post-processing
    (compiled with `nms_postprocess(meta_arch=yolov8, ...)`).

    Output format per class: [y_min, x_min, y_max, x_max, score]
    with coordinates normalised to [0, 1].
    """

    def __init__(self, hef_path: str = MODEL_PATH):
        """
        Load the HEF onto the Hailo device and prepare the
        inference pipeline.

        Args:
            hef_path: Path to the compiled .hef file.
        """
        if not os.path.isfile(hef_path):
            raise FileNotFoundError(
                f"HEF model not found at '{hef_path}'. "
                f"Ensure yolov8m.hef is in the Models/ directory."
            )

        print(f"[INFO] Loading HEF: {hef_path}")
        self.hef = HEF(hef_path)

        # Open a virtual device (auto-discovers the Hailo chip)
        self.vdevice = VDevice()

        # Configure the network group on the device
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]

        # Get stream info for input / output
        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        # Create VStream params (required by InferVStreams API)
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )

        # Read model input shape (usually [640, 640, 3])
        self.input_shape = self.input_vstream_info[0].shape
        self.input_h = self.input_shape[0]
        self.input_w = self.input_shape[1]

        # Number of classes from the output info
        self.num_classes = len(COCO_CLASSES)

        # ── Persistent pipeline (created ONCE, reused every frame) ────
        # Creating InferVStreams + activating the network per-frame
        # causes ~1.5-2 s overhead and kills FPS.  Keep them alive.
        self._open_pipeline()

        print(f"[INFO] Hailo device ready. Input shape: {self.input_shape}")
        print(f"[INFO] Output layers: {[o.name for o in self.output_vstream_info]}")
        print(f"[INFO] Persistent inference pipeline activated.")

        # Warm up the Hailo AI chip with a dummy inference.
        # This prevents a massive power-surge from Hailo compiling
        # its first pipeline concurrently with other hardware (like servos).
        print(f"[INFO] Warming up Hailo AI model with dummy inference...")
        dummy_input = np.zeros((1, self.input_h, self.input_w, 3), dtype=np.uint8)
        self.pipeline.infer({self.input_vstream_info[0].name: dummy_input})
        print(f"[INFO] AI warm-up complete.")

    # ── Pipeline management ────────────────────────────────────────────

    def _open_pipeline(self):
        """Create (or re-create) the persistent inference pipeline."""
        self._pipeline_ctx = InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        )
        self.pipeline = self._pipeline_ctx.__enter__()

        self._ng_ctx = self.network_group.activate()
        self._ng_ctx.__enter__()
        log.debug("[HAILO] Pipeline opened.")

    def _close_pipeline(self):
        """Tear down the inference pipeline (safe to call multiple times)."""
        for ctx in ("_ng_ctx", "_pipeline_ctx"):
            try:
                getattr(self, ctx).__exit__(None, None, None)
            except Exception:
                pass
        log.debug("[HAILO] Pipeline closed.")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize and format the frame to match the model's expected input.

        Args:
            frame: BGR image (numpy array).

        Returns:
            Preprocessed image (uint8, HxWxC).
        """
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        return resized

    def infer(self, preprocessed: np.ndarray) -> dict:
        """
        Run inference on the Hailo accelerator.

        IMPORTANT: The ThreadedCamera MUST be paused before calling
        this method, otherwise Picamera2's continuous capture_array()
        calls compete for system resources and cause a deadlock.
        The ``detect_objects()`` function handles this automatically.

        Args:
            preprocessed: Pre-processed image matching input shape.

        Returns:
            Raw output dict from the Hailo device.
        """
        input_data = {
            self.input_vstream_info[0].name:
                np.expand_dims(preprocessed, axis=0)
        }
        log.debug("[HAILO] pipeline.infer() start")
        res = self.pipeline.infer(input_data)
        log.debug("[HAILO] pipeline.infer() done")
        return res

    def close(self):
        """
        Tear down the persistent inference pipeline and release
        the Hailo device.  Call this when you are done with the model.
        """
        self._close_pipeline()
        print("[INFO] Hailo inference pipeline closed.")

    def postprocess(self, raw_output: dict,
                    orig_w: int, orig_h: int,
                    conf_threshold: float = CONFIDENCE_THRESHOLD) -> list:
        """
        Parse the NMS-postprocessed output from the Hailo model
        and return structured detection results.

        The HEF with on-chip NMS produces output as:
            tensor[batch][class_id] -> numpy array of shape (N, 5)
        where N varies per class (inhomogeneous), and each row is:
            [y_min, x_min, y_max, x_max, score]  (normalised 0–1)

        Args:
            raw_output:     Dict of output tensors from infer().
            orig_w:         Original frame width (for rescaling boxes).
            orig_h:         Original frame height (for rescaling boxes).
            conf_threshold: Minimum score to keep a detection.

        Returns:
            List of detection dicts (same format as detect_objects).
        """
        detections = []

        for layer_name, tensor in raw_output.items():
            # NMS-postprocessed output: tensor shape is (1, num_classes, ...)
            # where each class has a variable number of detections.
            # We CANNOT do np.array(tensor) because it's inhomogeneous.
            # Instead, iterate by class using Python indexing.

            # tensor[0] = first batch
            batch = tensor[0]
            num_classes = len(batch)

            for class_id in range(num_classes):
                class_dets = batch[class_id]

                # class_dets is a numpy array of shape (N, 5)
                # where N = number of detections for this class
                if class_dets is None or len(class_dets) == 0:
                    continue

                class_dets = np.array(class_dets)
                if class_dets.ndim == 1:
                    # Single detection: reshape to (1, 5)
                    class_dets = class_dets.reshape(1, -1)

                for det in class_dets:
                    if len(det) < 5:
                        continue

                    score = float(det[4])
                    if score < conf_threshold:
                        continue

                    y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]

                    # Scale normalised coords to original frame size
                    x1 = int(x_min * orig_w)
                    y1 = int(y_min * orig_h)
                    x2 = int(x_max * orig_w)
                    y2 = int(y_max * orig_h)

                    # Clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)

                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue

                    label = (COCO_CLASSES[class_id]
                             if class_id < len(COCO_CLASSES)
                             else f"class_{class_id}")

                    detections.append({
                        'label': label,
                        'class_id': class_id,
                        'confidence': score,
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'width_px': w,
                        'height_px': h,
                    })

        return detections


# ===================================================================
# Standalone Detection Function (reusable)
# ===================================================================

def detect_objects(model: HailoYOLOv8, frame: np.ndarray,
                   conf_threshold: float = CONFIDENCE_THRESHOLD,
                   camera=None) -> list:
    """
    Run YOLOv8 inference on the Hailo accelerator and return
    structured detection results.

    Args:
        model:          HailoYOLOv8 instance.
        frame:          BGR image (numpy array).
        conf_threshold: Minimum confidence to keep a detection.
        camera:         Optional ThreadedCamera – will be paused during
                        inference to prevent Picamera2 / Hailo contention.

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
    log.debug("[DETECT] Starting preprocessing")
    orig_h, orig_w = frame.shape[:2]
    preprocessed = model.preprocess(frame)

    # Pause camera thread to prevent GIL / DMA contention with Hailo
    if camera is not None and hasattr(camera, 'pause'):
        camera.pause()

    log.debug("[DETECT] Preprocessing complete. Starting infer()")
    try:
        raw_output = model.infer(preprocessed)
    finally:
        # Always resume camera, even if inference fails
        if camera is not None and hasattr(camera, 'resume'):
            camera.resume()

    log.debug("[DETECT] infer() complete. Starting postprocess()")
    detections = model.postprocess(raw_output, orig_w, orig_h, conf_threshold)
    log.debug("[DETECT] postprocess() complete. %d detections.", len(detections))
    return detections


# ===================================================================
# Depth Estimation (reusable)
# ===================================================================

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
        Estimated depth in centimetres. Returns -1 if calculation
        is not possible (zero-width box).
    """
    if bbox_width_px <= 0:
        return -1.0
    depth_cm = (known_width_cm * focal_length_px) / bbox_width_px
    return round(depth_cm, 1)


# ===================================================================
# Drawing Functions (reusable)
# ===================================================================

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

    info = f"FPS: {fps:.1f}  |  Objects: {num_detections}  |  Hailo AI HAT+"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

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
                    #print("[DEBUG-CAMERA] About to call picam2.capture_array()")
                    frame = self.picam2.capture_array()
                    #print("[DEBUG-CAMERA] capture_array() returned")
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


class ThreadedCamera:
    """
    Constantly reads frames from the camera in a background thread.
    This prevents buffer exhaustion (which causes libcamera/picamera2 to hang)
    when the main thread is busy doing inference or sleeping for servos.

    Call ``pause()`` before Hailo inference and ``resume()`` after – the
    continuous capture_array() calls from this thread compete for system
    resources (GIL / DMA) and cause the Hailo pipeline to deadlock.
    """
    def __init__(self, camera):
        self.camera = camera
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.last_update_time = time.time()

        # Event is SET when the thread should be actively reading.
        # CLEAR it to pause camera reads (e.g. during Hailo inference).
        self._run_event = threading.Event()
        self._run_event.set()      # start in "running" state

        # Read the first frame to ensure it's working
        self.ret, self.frame = self.camera.read()
        if self.ret:
            self.last_update_time = time.time()

        # Start the background thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print("[INFO] ThreadedCamera background stream started.")

    # ── pause / resume ────────────────────────────────────────────────

    def pause(self):
        """Pause background frame capture (call before Hailo infer)."""
        self._run_event.clear()
        # Wait briefly for any in-flight capture_array() to finish
        time.sleep(0.05)

    def resume(self):
        """Resume background frame capture (call after Hailo infer)."""
        self._run_event.set()

    # ── background loop ───────────────────────────────────────────────

    def _update(self):
        while self.running:
            # Block here while paused (no camera reads during inference)
            self._run_event.wait()
            try:
                ret, frame = self.camera.read()
            except Exception as e:
                print(f"[WARN] ThreadedCamera read exception: {e}")
                ret, frame = False, None

            with self.lock:
                self.ret = ret
                if ret and frame is not None:
                    self.frame = frame
                    self.last_update_time = time.time()
            # Small sleep to yield CPU slightly, though read() is blocking.
            time.sleep(0.001)

    def read(self):
        with self.lock:
            # If we haven't received a new frame in 2 seconds, consider the stream dead/frozen
            if time.time() - self.last_update_time > 2.0:
                self.ret = False

            # Return a copy to avoid race conditions if the drawing thread modifies it
            frame = self.frame.copy() if self.frame is not None else None
            return self.ret, frame

    def release(self):
        self.running = False
        self._run_event.set()      # unblock if paused so thread can exit
        self.thread.join(timeout=1.0)
        self.camera.release()
        print("[INFO] ThreadedCamera stream stopped.")

    def isOpened(self):
        return self.camera.isOpened()


def open_camera(camera_index: int = 0, width: int = 640, height: int = 480):
    """
    Try every available camera backend in priority order and return
    the first one that works, wrapped in a ThreadedCamera to prevent hangs.

    Priority:
        1. Picamera2  (native CSI – IMX219, etc.)
        2. GStreamer libcamera pipeline
        3. OpenCV V4L2 / generic (USB webcams)

    Returns:
        A ThreadedCamera object with .read(), .release(), .isOpened() methods,
        or None if nothing works.
    """
    print("[INFO] Probing camera backends...")

    cap = _try_picamera2(width, height)
    if not cap:
        cap = _try_gstreamer_libcamera(width, height)
    if not cap:
        cap = _try_opencv_v4l2(camera_index, width, height)

    if cap:
        return ThreadedCamera(cap)
        
    return None


# ===================================================================
# Real-time Test Loop
# ===================================================================

def run_realtime_test(camera_index: int = 0):
    """
    Open a camera feed (CSI or USB), run YOLOv8m detection on the
    Hailo AI HAT+, and display the annotated output in a window.

    Press 'q' to quit.

    Args:
        camera_index: Index of the camera device (used for USB fallback).
    """
    # -------------------------------------------
    # 1. Load the Hailo model
    # -------------------------------------------
    model = HailoYOLOv8(MODEL_PATH)

    # -------------------------------------------
    # 2. Open camera (auto-detect best backend)
    # -------------------------------------------
    cap = open_camera(camera_index, width=640, height=480)
    if cap is None:
        print("[ERROR] Could not open camera on any backend.")
        print("        Ensure Picamera2 is installed:  pip install picamera2")
        return

    print("[INFO] Camera ready. Press 'q' to quit.")
    print("=" * 60)

    prev_time = time.time()
    fps = 0.0

    window_name = "YOLOv8m - Bounding Box & Depth Test (Hailo)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame. Retrying...")
            time.sleep(0.1)
            continue

        # --- Detection on Hailo AI HAT+ ---
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
                depth_str = (f"{d.get('depth_cm', '?')}cm"
                             if d.get('depth_cm', -1) > 0 else "N/A")
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
