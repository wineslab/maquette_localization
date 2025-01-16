import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# QR Code detector
qr_code_detector = cv2.QRCodeDetector()

# Function to save a frame as a PNG file
def save_frame_as_png(frame, filename="frame.png"):
    cv2.imwrite(filename, frame)

# Test saving a frame
ret, test_frame = cap.read()
if ret:
    save_frame_as_png(test_frame, "test_frame.png")
else:
    print("Failed to grab a test frame")
exit()





while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect QR Code in the frame
    data, bbox, _ = qr_code_detector.detectAndDecode(frame)

    if bbox is not None:
        # Draw bounding box around the detected QR code
        bbox = np.int32(bbox)
        for i in range(len(bbox[0])):
            cv2.line(frame, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % len(bbox[0])]), (255, 0, 0), 2)

        if data:
            # Display the data
            print("QR Code Data:", data)

            # Calculate the center of the QR code
            center_x = int(np.mean(bbox[0][:, 0]))
            center_y = int(np.mean(bbox[0][:, 1]))
            print(f"QR Code Center: ({center_x}, {center_y})")

            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("QR Code Localization", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
