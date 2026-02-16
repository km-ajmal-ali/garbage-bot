import os
import numpy as np
from PIL import Image

# Settings
IMAGE_DIR = './Dataset/train/images' # Path to your training images
OUTPUT_FILE = 'calib_set.npy'
IMAGE_SIZE = (640, 640)
NUM_IMAGES = 64

def create_calibration_set():
    calib_data = []
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(files)} images. Processing {NUM_IMAGES}...")

    for i in range(min(NUM_IMAGES, len(files))):
        img_path = os.path.join(IMAGE_DIR, files[i])
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1] if required, 
        # or leave as [0, 255] depending on your ONNX export settings.
        # YOLOv8/11 usually expects 0-255 but check your specific export.
        img_array = np.array(img).astype(np.float32)
        calib_data.append(img_array)

    calib_set = np.array(calib_data)
    np.save(OUTPUT_FILE, calib_set)
    print(f"Calibration set saved to {OUTPUT_FILE} with shape {calib_set.shape}")

if __name__ == "__main__":
    create_calibration_set()