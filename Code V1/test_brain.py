import moondream as md
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the model using the local weights file
# model = md.vl(model="./moondream-0_5b-int8.mf")

# Use 'cpu' instead of 'cuda'
model_id = "moondream/moondream-2b-2025-04-14-4bit"

print("Loading 2B model on CPU... (This will be slow)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map={"": "cpu"}, # Force CPU usage
    torch_dtype=torch.float32 # Pi 5 handles float32 best
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load an image (make sure you have an image file in the same folder)
image = Image.open("camera_shot.jpeg")

# Query the model
answer = model.query(image, "What type of garbage do you see?")["answer"]
print("Brain Response:", answer)