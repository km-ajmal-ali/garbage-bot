import cv2
import numpy as np
img = np.zeros((400, 400, 3), dtype=np.uint8)
det = cv2.QRCodeDetector()
ret, pts = det.detect(img)
print("detect:", ret, type(pts), pts.shape if pts is not None else None)

# Make a dummy QR code to detect
img[100:300, 100:300] = 255
# wait, det needs a real qr code to return True
