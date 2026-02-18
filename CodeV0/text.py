import time
from VisualModel.detector import HailoDetector

# Initialize
# Note: The HEF path assumes the script is run from the CodeV0 directory
detector = HailoDetector("VisualModel/yolo11n.hef")

def detection_loop():
    print("Detection Loop Started: Press Ctrl+C to stop.")
    print("Visual output should appear in the 'WasteBot View' window.")
    
    while True:
        # STEP 1: SCANNING PHASE
        # detector.get_objects() captures the frame, runs detection, 
        # draws bounding boxes on the frame, and displays the "WasteBot View" window/
        detections = detector.get_objects() # Returns list of {'label': 'waste', 'x': 320, 'w': 50}
        
        # Output detections to console for verification
        if detections:
            print(f"Detections: {detections}")
        
        # The detector loop runs based on camera FPS. 
        # detector.get_objects() includes cv2.waitKey(1) internally.

if __name__ == "__main__":
    try:
        detection_loop()
    except KeyboardInterrupt:
        print("Detection Stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
