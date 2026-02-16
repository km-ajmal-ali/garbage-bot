from ultralytics import YOLO

# 1. Load your trained model weights
# Replace "best.pt" with the path to your actual trained weights
model = YOLO("./runs/detect/train3/weights/best.pt")

# 2. Export to ONNX with precise Hailo-compatible parameters
model.export(
    format="onnx", 
    imgsz=640, 
    dynamic=False,   # Hailo requires static shapes for optimization
    opset=11,        # Most stable for Hailo DFC 3.33.0 parsing
    simplify=True,   # Removes unnecessary ONNX nodes that confuse the compiler
    end2end=False    # CRITICAL: Prevents the "expected conv but found concat" error
)

print("Export complete. Your 'clean' ONNX is ready for the Hailo DFC.")