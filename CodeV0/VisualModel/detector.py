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
            self.cap = cv2.VideoCapture(0)
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
            return []

        # 1. Detect QR Codes
        detections = []
        data, bbox, _ = self.qr_detector.detectAndDecode(frame)
        if data and bbox is not None:
            # bbox is usually [[x1,y1], [x2,y2], ...]
            x, y, w, h = self.cv2.boundingRect(bbox)
            detections.append({'label': 'qr', 'x': x + w//2, 'w': w})

        # 2. Detect Waste (YOLO)
        raw_output = self.detect(frame)
        # Parse YOLO output (Assuming shape [1, 5, 8400] for 1 class)
        # This is a simplified parser. For robust use, NMS is needed.
        for name, tensor in raw_output.items():
            # tensor shape (1, 5, 8400) -> 5 rows: x, y, w, h, conf
            # We transpose to (8400, 5) to iterate
            preds = tensor[0].T 
            
            # Simple threshold filtering
            confidence_threshold = 0.5
            start_idx = 4 # 0=x, 1=y, 2=w, 3=h, 4=conf (if 1 class)
            
            # Filter by confidence
            # Note: This index logic assumes specific YOLO export format.
            # If classes > 1, preds would be 4 + num_classes
            
            # Check for high confidence detections
            high_conf = preds[preds[:, 4] > confidence_threshold]
            
            for det in high_conf:
                x, y, w, h, conf = det[:5]
                # Scale coordinates back to original frame size if needed
                # (Training was 640, Camera usually 640x480 or 1080p)
                # Here we assume resize in detect() aligns or we just return 640-scale coords
                # main.py logic handles x < 300 (center of 640 is 320), so 640 scale is good.
                
                detections.append({
                    'label': 'waste',
                    'x': int(x),
                    'w': int(w)
                })
                
                # Limit to one detection for now to avoid noise flood without NMS
                # Or keep reliable ones. 
                break # Just return the strongest one per frame for simplicity
        
        return detections