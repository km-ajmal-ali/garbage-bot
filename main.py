import cv2
import time
import moondream as md
from PIL import Image
from Common.motors import MotorControl
from Common.servos import CameraServo

# Init hardware and brain
# We use the tiny 0.5b model for 'best' speed on Pi 5
model = md.vl(model="./moondream-0_5b-int8.mf") 
motors = MotorControl(pins=[17, 18, 22, 23, 12, 13])
eye = CameraServo(pin=25)
cam = cv2.VideoCapture(0)

def capture_and_analyze(prompt):
    ret, frame = cam.read()
    if not ret: return None
    # Convert OpenCV BGR to PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    encoded_image = model.encode_image(image)
    return model.query(encoded_image, prompt)["answer"].lower()

def mission_control():
    while True:
        # STEP 1: SCAN FOR WASTE
        print("Brain: Looking for waste...")
        answer = capture_and_analyze("Is there any waste or garbage on the floor? Answer Yes or No.")
        
        if "yes" in answer:
            # STEP 2: MOVE TO WASTE
            # Ask for direction with displacement
            nav = capture_and_analyze("Where is the waste? Answer: 'left', 'right', or 'center'.")
            print(f"Brain: Waste found at {nav}")
            
            if "left" in nav: motors.move("left", speed=30); time.sleep(0.5)
            elif "right" in nav: motors.move("right", speed=30); time.sleep(0.5)
            else: motors.move("forward", speed=50); time.sleep(1.0)
            
            # Check if close enough
            dist = capture_and_analyze("Is the waste within reach (less than 10cm)? Yes or No.")
            if "yes" in dist:
                motors.stop()
                print("Action: Collecting waste...")
                time.sleep(3) # Pickup simulation
        
        else:
            # STEP 3: SEARCH FOR QR CODE
            print("Brain: No waste. Searching for QR Home Station...")
            qr_check = capture_and_analyze("Do you see a QR code? Yes or No.")
            
            if "yes" in qr_check:
                motors.move("forward", speed=40)
                print("Brain: Returning to base.")
            else:
                eye.look_around() # Pan camera to search
                motors.move("left", speed=30); time.sleep(0.3) # Pivot car

if __name__ == "__main__":
    mission_control()