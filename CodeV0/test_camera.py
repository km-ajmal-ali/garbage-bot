
import cv2
import sys

def check_cameras():
    print("Checking for available cameras...")
    
    # Check if we are on Windows or Linux
    is_windows = sys.platform.startswith('win')
    
    # List of backends to try
    backends = []
    if is_windows:
        backends.append((cv2.CAP_DSHOW, "DirectShow"))
        backends.append((cv2.CAP_MSMF, "Media Foundation"))
    else:
        backends.append((cv2.CAP_V4L2, "V4L2"))
        backends.append((cv2.CAP_GSTREAMER, "GStreamer"))
    
    backends.append((cv2.CAP_ANY, "Any/Auto"))

    # Try indices 0 to 5
    found_any = False
    for index in range(5):
        print(f"\n--- Checking Camera Index {index} ---")
        for backend_id, backend_name in backends:
            cap = cv2.VideoCapture(index, backend_id)
            if cap.isOpened():
                print(f"  [SUCCESS] Opened with backend: {backend_name}")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    print(f"    Frame read successful! Resolution: {w}x{h}")
                    found_any = True
                else:
                    print(f"    Opened, but failed to read frame.")
                
                cap.release()
            else:
                # print(f"  [FAILED] Could not open with backend: {backend_name}")
                pass

    # Additionally check specific GStreamer pipeline for RPi
    if not is_windows:
        print("\n--- Checking RPi Libcamera (GStreamer) ---")
        pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"  [SUCCESS] Opened with GStreamer pipeline")
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"    Frame read successful! Resolution: {w}x{h}")
                found_any = True
            else:
                print(f"    Opened, but failed to read frame.")
            cap.release()
        else:
            print("  [FAILED] Could not open GStreamer pipeline")

    if not found_any:
        print("\nNo working cameras found on tested configurations.")

if __name__ == "__main__":
    check_cameras()
