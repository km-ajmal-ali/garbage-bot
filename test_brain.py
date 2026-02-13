import moondream as md
from PIL import Image

# Initialize the model using the local weights file
model = md.vl(model="./moondream-0_5b-int8.mf")

# Load an image (make sure you have an image file in the same folder)
image = Image.open("camera_shot.jpg")

# Query the model
answer = model.query(image, "What type of garbage do you see?")["answer"]
print("Brain Response:", answer)