from ultralytics import YOLO

# 1. Load a pretrained Nano model (fastest for RPi5 + Hailo)
model = YOLO("yolo11n.pt") 

# 2. Train the model
# You need a 'data.yaml' file pointing to your garbage + QR images
results = model.train(
    data="garbage_data.yaml", 
    epochs=100, 
    imgsz=640, 
    device=0  # Uses your GTX 1650
)

# 3. Export to ONNX (Requirement for Hailo Dataflow Compiler)
model.export(format="onnx")

print("Training complete. Now use Hailo DFS Tool to convert ONNX to HEF.")
