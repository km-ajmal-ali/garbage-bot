import os
import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

# 1. Configuration
model_name = "mb_garbage"
onnx_path = "best.onnx"          # Path to your exported YOLOv8 ONNX file
har_path = f"{model_name}.har"
hef_path = f"{model_name}.hef"
calib_path = "../dataset/train/images" # Folder with ~64 images from your dataset
batch_size = 1

# 2. Hardware Architecture
# Use 'hailo8' for the 26 TOPS AI HAT+
target_arch = "hailo8" 

def run_conversion():
    # Initializing the Hailo Runner
    runner = ClientRunner(hw_arch=target_arch)

    # STEP 1: Translate ONNX to Hailo Archive (HAR)
    print("Translating ONNX to HAR...")
    runner.translate_onnx_model(
        onnx_path, 
        model_name,
        start_node_names=['images'], # Standard for YOLOv8
        end_node_names=None          # DFC usually detects YOLOv8 heads automatically
    )
    runner.save_har(har_path)

    # STEP 2: Optimization (Quantization)
    print("Starting Optimization/Quantization...")
    # Load calibration images and preprocess them (resize to 640x640)
    def load_calib_data(folder):
        images = []
        for img_name in os.listdir(folder)[:64]: # Hailo recommends 64-100 images
            img = Image.open(os.path.join(folder, img_name)).resize((640, 640))
            images.append(np.array(img))
        return np.array(images)

    calib_dataset = load_calib_data(calib_path)
    
    # Apply quantization (converts float32 to int8)
    runner.optimize(calib_dataset)

    # STEP 3: Compilation
    print(f"Compiling for {target_arch}...")
    # This step generates the HEF binary for the 26 TOPS hardware
    hef = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef)
    
    print(f"Success! HEF file saved as: {hef_path}")

if __name__ == "__main__":
    run_conversion()