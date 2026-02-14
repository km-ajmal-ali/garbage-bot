import time
from VisualModel.detector import HailoDetector
# from Brain.moondream_brain import MoondreamBrain
from Local.motors import MotorControl
from Local.servos import CameraServo

# Initialize
detector = HailoDetector("garbage_qr_model.hef")
# brain = MoondreamBrain("moondream.bin")
motors = MotorControl(pins=[17, 18, 22, 23, 12, 13])
eye = CameraServo(pin=25)

def mission_control():
    print("Mission Started: Searching for waste...")
    
    while True:
        # STEP 1: SCANNING PHASE
        detections = detector.get_objects() # Returns list of {'label': 'waste', 'x': 320, 'w': 50}
        
        waste = [d for d in detections if d['label'] == 'waste']
        qr_code = [d for d in detections if d['label'] == 'qr']

        if waste:
            # STEP 2: FETCHING PHASE
            target = waste[0] # Pick the first waste found
            print(f"Waste detected! Moving to center: {target['x']}")
            
            # Simple Proportional Control for steering
            if target['x'] < 300: motors.move("left", speed=40)
            elif target['x'] > 340: motors.move("right", speed=40)
            else: motors.move("forward", speed=60)
            
            # Depth check: if bounding box width > 400px, it's close enough to pick up
            if target['w'] > 400:
                motors.stop()
                print("Collecting waste...")
                time.sleep(2) # Simulate collection mechanism
                
        elif not waste and qr_code:
            # STEP 3: RETURN TO BASE PHASE
            print("No waste left. Returning to QR Docking Station.")
            target = qr_code[0]
            # Similar steering logic as above to move toward QR
            if target['w'] > 300:
                print("Docked at QR. Mission Complete.")
                break
                
        else:
            # STEP 4: SEARCH PHASE (Look Around)
            print("Nothing seen. Rotating camera...")
            eye.look_around() 
            motors.move("left", speed=30) # Rotate car slowly to find targets
            time.sleep(0.5)
            motors.stop()

if __name__ == "__main__":
    try:
        mission_control()
    except KeyboardInterrupt:
        motors.stop()