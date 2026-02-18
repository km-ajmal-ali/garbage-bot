import cv2
import glob
import os

def check_camera_indices():
    # Find all /dev/video* devices
    devices = glob.glob('/dev/video*')
    print(f"Found video devices: {devices}")
    
    # Try indices 0 to 10
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"\nChecking Camera Index {i}:")
            backend = cap.getBackendName()
            print(f"  Backend: {backend}")
            
            # Read a frame without setting anything
            ret, frame = cap.read()
            if ret:
                print("  Success: Captured frame!")
                print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("  Failed to capture frame (default settings).")
            
            # Releases
            cap.release()
            
            # Try with MJPG settings as in detector.py
            print(f"  Retrying Index {i} with MJPG settings...")
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret, frame = cap.read()
            if ret:
                print("  Success: Captured frame with MJPG!")
            else:
                print("  Failed to capture frame with MJPG.")
            cap.release()
        else:
            pass # Index not available

if __name__ == "__main__":
    check_camera_indices()
