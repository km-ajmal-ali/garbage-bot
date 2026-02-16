Hailo Software Suite: Download the Docker image (usually a .tar file) from the Hailo Developer Zone.

# Load the image if you haven't yet
docker load -i hailo_ai_sw_suite_2024-10.tar

# Run the container with a local volume
docker run -it --rm -v "C:\path\to\your\model_folder:/workspace" hailo_ai_sw_suite_2024-10:1


# Hailo Dataflow Compiler (DFC) Conversion Guide

**Target Hardware:** Hailo-8L (Raspberry Pi 5 AI Kit)

**Model:** YOLO11n

**DFC Version:** 3.33.0

This document outlines two paths: **Standard Compilation** (basic conversion) and **Optimized Compilation** (high-performance detection using the NPU for post-processing).

---

## 1. Environment Preparation

Before running commands, ensure you are in the correct directory where your Windows files are mounted and activate your virtual environment.

```bash
# Move to the directory linked to your Windows E: drive
cd /workspace

# Activate the pre-configured Hailo environment
source hailo_virtualenv/bin/activate

```

---

## 2. Path A: Standard Compilation (No `.alls` script)

Use this if you want a quick conversion. The Raspberry Pi CPU will handle all box decoding (NMS), which is easier to set up but results in lower FPS.

### Step 1: Parse ONNX to HAR

Translates the ONNX graph into Hailo's internal archive format.

```bash
hailo parser onnx yolo11n.onnx --hw-arch hailo8l

```

### Step 2: Optimize (Quantize)

Converts the model from Float32 to INT8. Note: Without a calibration set, accuracy will be poor.

```bash
hailo optimize yolo11n.har --hw-arch hailo8l

```

### Step 3: Compile to HEF

Generates the final binary for the Pi.

```bash
# In DFC 3.33.0, use --output-dir to specify destination
hailo compiler yolo11n.har --hw-arch hailo8l --output-dir .

```

---

## 3. Path B: Optimized Compilation (With `.alls` script)

**Recommended for WasteBot.** This path bakes the Non-Maximum Suppression (NMS) logic into the Hailo chip to maximize FPS.

### Step 1: Create the Calibration Dataset

Run this Python script to create `calib_set.npy` from your training images.

```python
import numpy as np
import os
from PIL import Image

# Use ~64-128 images from your dataset
img_path = "./Dataset/train/images"
images = []
for f in os.listdir(img_path)[:64]:
    img = Image.open(os.path.join(img_path, f)).resize((640, 640))
    images.append(np.array(img))

np.save("calib_set.npy", np.array(images))

```

### Step 2: Create the Model Script (`yolo11n.alls`)

Create a text file named `yolo11n.alls`. This tells the compiler to handle the detection logic.

```text
# Sets optimization level
model_optimization_flavor(optimization_level=2, compression_level=0)

# Configures NMS (Post-processing) for DFC 3.33.0 syntax
# Note: YOLO11 uses 'yolov8' meta_arch
nms_postprocess(meta_arch=yolov8, nms_scores_th=0.3, nms_iou_th=0.45, engine=auto)

# Maximizes compiler effort
performance_param(compiler_optimization_level=max)

```

### Step 3: The Optimized Pipeline

```bash
# 1. Parse
hailo parser onnx yolo11n.onnx --hw-arch hailo8l

#1, a) Needed to disabled recomended run, if no the run fails
hailo parser onnx yolo11n.onnx --hw-arch hailo8l \
--end-node-names /model.23/cv2.0/cv2.0.2/Conv /model.23/cv3.0/cv3.0.2/Conv \
                 /model.23/cv2.1/cv2.1.2/Conv /model.23/cv3.1/cv3.1.2/Conv \
                 /model.23/cv2.2/cv2.2.2/Conv /model.23/cv3.2/cv3.2.2/Conv

# 2. Optimize with Script and Calibration Data
hailo optimize yolo11n.har \
    --calib-set-path calib_set.npy \
    --model-script yolo11n.alls \
    --hw-arch hailo8l

# 3. Compile
hailo compiler yolo11n.har --hw-arch hailo8l --output-dir .

```

---

## 4. Troubleshooting & Tips

| Error | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError` | Terminal is in `/local/workspace` | Run `cd /workspace` first. |
| `unrecognized arguments: --output-path` | CLI version difference | Use `--output-dir .` instead. |
| `No argument named nms_score_threshold` | Version 3.33.0 syntax | Use `nms_scores_th` and `nms_iou_th` in `.alls`. |
| `Permission Denied` on `/workspace` | Docker/Windows mount sync | Run `chmod -R 777 /workspace` from host or container. |

---

## 5. Deployment on Raspberry Pi 5

Once you have `yolo11n.hef`, move it to your Pi and run a quick benchmark:

```bash
hailortcli benchmark yolo11n.hef

```