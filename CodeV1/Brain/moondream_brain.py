import moondream as md
from PIL import Image

class MoondreamBrain:
    def __init__(self, model_path):
        self.model = md.load(model_path) # Load the 0.5B model for speed

    def analyze_scene(self, image_path, prompt):
        image = Image.open(image_path)
        # Asking Moondream for strategic advice
        answer = self.model.answer_question(image, prompt)
        return answer