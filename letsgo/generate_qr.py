"""
generate_qr.py
==============
Generates a simple QR code image that the bot can detect.
"""
import qrcode

def generate():
    data = "WasteBot Destination"
    
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=20,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save it
    filename = "destination_qr.png"
    img.save(filename)
    print(f"Generated {filename} successfully! Print this at ~15x15 cm for best results.")

if __name__ == "__main__":
    generate()
