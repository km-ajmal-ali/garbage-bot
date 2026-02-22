from hailo_sdk_client import ClientRunner
import numpy as np

# 1. Initialize the runner (no arguments)
runner = ClientRunner()

# 2. Load the HAR file
har_path = "model.har"
runner.load_har(har_path)
print(f"Successfully loaded {har_path}")

# 3. Load your .npy file
calib_data = np.load("calib_set.npy")
# Use the input node name found in your previous logs
input_node_name = 'yolov11n/input_layer1'
data_dict = {input_node_name: calib_data}

print(f"Loaded calibration data with shape: {calib_data.shape}")

# 4. Optimization (Quantization)
# This may take a few minutes on CPU
runner.optimize(data_dict)

# 5. Compile for Hailo-8
hef = runner.compile()

# 6. Save the final file
output_name = "yolo11n_rpi_ai_hat.hef"
with open(output_name, "wb") as f:
    f.write(hef)

print(f"Final .hef file created successfully: {output_name}")