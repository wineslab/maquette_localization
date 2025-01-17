import cv2
import numpy as np

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1920 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 1080 pixels

# QR Code detector
qr_code_detector = cv2.QRCodeDetector()

# Store detected QR codes and their positions
detected_qr_codes = {}

def preprocess_frame(frame):
    """
    Preprocess the frame to enhance QR code detection:
    - Convert to grayscale
    - Apply Gaussian blur
    - Apply adaptive thresholding
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply slight blurring
    _, threshold = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)  # Apply thresholding
    return threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame for better QR detection
    preprocessed_frame = preprocess_frame(frame)

    # Multi-scale detection for better accuracy
    scales = [1.0, 1.5, 2.0]  # Scale factors to test
    retval, decoded_info, points, straight_qrcode = None, None, None, None

    for scale in scales:
        resized_frame = cv2.resize(preprocessed_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        retval, decoded_info, points, straight_qrcode = qr_code_detector.detectAndDecodeMulti(resized_frame)
        if retval:
            break

    if retval:  # If QR codes are detected
        points = np.int32(points)

        # Loop through all detected QR codes
        for i, box in enumerate(points):
            qr_data = decoded_info[i]

            # Calculate center of the QR code
            center_x = int(np.mean(box[:, 0]))
            center_y = int(np.mean(box[:, 1]))

            if qr_data:
                # Store QR code data and its center position
                detected_qr_codes[qr_data] = (center_x, center_y)

                # Draw bounding box and label
                for j in range(4):
                    cv2.line(frame, tuple(box[j]), tuple(box[(j + 1) % 4]), (255, 0, 0), 2)

                cv2.putText(frame, qr_data, (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Calculate relative positions
        if detected_qr_codes:
            reference_qr = next(iter(detected_qr_codes))  # Use the first detected QR as the reference
            ref_x, ref_y = detected_qr_codes[reference_qr]

            for qr, (x, y) in detected_qr_codes.items():
                if qr != reference_qr:
                    rel_x = x - ref_x
                    rel_y = y - ref_y
                    print(f"{qr} relative to {reference_qr}: ({rel_x}, {rel_y})")

                    # Display relative position on the frame
                    cv2.putText(frame, f"Rel: ({rel_x}, {rel_y})", (x, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Resize the display frame to fit on the screen
    display_frame = cv2.resize(frame, (960, 540))  # Resize to 960x540 for display
    cv2.imshow("QR Code Relative Localization", display_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
