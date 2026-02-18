import numpy as np
from hailo_platform import HEF, Device, VDevice, HailoStreamInterface, InferVStreams

class HailoDetector:
    def __init__(self, hef_path):
        self.device = VDevice()
        self.hef = HEF(hef_path)
        self.network_group = self.device.configure(self.hef)[0]
        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()
        
        # Initialize Camera
        try:
            import cv2
            self.cv2 = cv2
            # Initialize Camera with Robust Fallback
            self.cv2 = cv2
            self.cap = None
            
            def try_open_camera(index, backend, use_mjpg):
                print(f"Attempting to open camera index {index} with backend {backend} (MJPG={use_mjpg})...")
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    return None
                
                if use_mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Try to read a frame to verify it works
                ret, _ = cap.read()
                if not ret:
                    print("  -> device opened but failed to read frame.")
                    cap.release()
                    return None
                print("  -> Camera opened and reading successfully.")
                return cap

            # Define a list of potential camera configurations to test
            # Format: (source, backend, use_mjpg, description)
            configs = [
                # 1. Try GStreamer pipeline for RPi (libcamera)
                ("libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink", getattr(cv2, 'CAP_GSTREAMER', 1800), False, "GStreamer (libcamera)"),
                # 2. Try V4L2 (Linux/RPi legacy) - Index 0
                (0, getattr(cv2, 'CAP_V4L2', 200), True, "V4L2 Index 0 (MJPG)"),
                (0, getattr(cv2, 'CAP_V4L2', 200), False, "V4L2 Index 0 (Default)"),
                # 3. Try DirectShow (Windows) - Index 0 and 1
                (0, getattr(cv2, 'CAP_DSHOW', 700), False, "DirectShow Index 0"),
                (1, getattr(cv2, 'CAP_DSHOW', 700), False, "DirectShow Index 1"),
                # 4. Try Generic/Any - Index 0 and 1
                (0, cv2.CAP_ANY, False, "Any Index 0"),
                (1, cv2.CAP_ANY, False, "Any Index 1"),
            ]

            for source, backend, mjpg, desc in configs:
                self.cap = try_open_camera(source, backend, mjpg)
                if self.cap:
                    print(f"Success: Initialized camera using {desc}")
                    break


            if self.cap is None:
                print("Error: Could not initialize camera on any tested configuration.")print("Error: Could not initialize camera on any tested configuration.")
            else:
                self.qr_detector = cv2.QRCodeDetector()
        except ImportError:
            print("OpenCV not installed. Camera functions will fail.")
            self.cap = None


    def detect(self, frame):
        # Pre-process frame to match model input (e.g., 640x640)
        resized_frame = self.cv2.resize(frame, (640, 640))
        
        # Expand dims for batch and run inference
        with InferVStreams(self.network_group, self.input_vstream_info, self.output_vstream_info) as infer_pipeline:
            input_data = {self.input_vstream_info[0].name: np.expand_dims(resized_frame, axis=0)}
            with self.network_group.activate_context():
                output_data = infer_pipeline.infer(input_data)
        
        return output_data

    def get_objects(self):
        """
        Captures a frame, runs detection, and returns formatted results.
        Returns: list of dicts {'label': str, 'x': int, 'w': int}
        """
        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized.")
            return []

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera")
            return []

        # Resize to 640x640 for consistent processing and display
        frame = self.cv2.resize(frame, (640, 640))

        # 1. Detect QR Codes
        detections = []
        data, bbox, _ = self.qr_detector.detectAndDecode(frame)
        if data and bbox is not None:
            # bbox is usually [[x1,y1], [x2,y2], ...]
            # Convert to int for drawing
            if bbox is not None:
                bbox_ints = bbox.astype(int)
                x, y, w, h = self.cv2.boundingRect(bbox_ints)
                detections.append({'label': 'qr', 'x': x + w//2, 'w': w})
                
                # Draw QR Box
                self.cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                self.cv2.putText(frame, f"QR: {data}", (x, y-10), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Detect Waste (YOLO)
        raw_output = self.detect(frame)
        
        # Parse YOLO output
        for name, tensor in raw_output.items():
            # tensor shape (1, 5, 8400) -> 5 rows: x, y, w, h, conf
            preds = tensor[0].T 
            
            # Simple threshold filtering
            confidence_threshold = 0.5
            start_idx = 4 
            
            # Check for high confidence detections
            high_conf = preds[preds[:, 4] > confidence_threshold]
            
            for det in high_conf:
                x, y, w, h, conf = det[:5]
                
                # Visualize
                # YOLO output is usually center_x, center_y, w, h
                x_tl = int(x - w / 2)
                y_tl = int(y - h / 2)
                w_px = int(w)
                h_px = int(h)
                
                self.cv2.rectangle(frame, (x_tl, y_tl), (x_tl + w_px, y_tl + h_px), (0, 0, 255), 2)
                self.cv2.putText(frame, f"Waste: {conf:.2f}", (x_tl, y_tl-10), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                detections.append({
                    'label': 'waste',
                    'x': int(x),
                    'w': int(w)
                })
        
        # Show what the camera sees
        self.cv2.imshow("WasteBot View", frame)
        self.cv2.waitKey(1)
        
        return detections