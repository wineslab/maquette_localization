import qrcode

# Create and save QR codes
for i in range(1, 6):  # Generate 5 QR codes with unique IDs
    data = f"QR_{i}"  # Unique identifier for each QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(f"QR_{i}.png")  # Save each QR code as an image file
