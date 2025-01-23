import cv2
import numpy as np
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define color ranges in HSV (tweak these values based on your markers)
color_ranges = {
    "red": ((0, 120, 70), (10, 255, 255)),  # Lower and upper HSV bounds for red
    "green": ((40, 40, 40), (80, 255, 255)),  # Lower and upper HSV bounds for green
    "blue": ((100, 150, 0), (140, 255, 255)),  # Lower and upper HSV bounds for blue
}

# Create output directory if not exists
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

# Function to detect color and return center positions
def detect_color_positions(frame, color_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV
    positions = {}

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))  # Create mask for the color
        mask = cv2.erode(mask, None, iterations=2)  # Remove noise
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour (assume it's the marker)
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 20:  # Minimum area to avoid small noise
                # Calculate the center of the contour
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    positions[color] = (center_x, center_y)

                    # Draw the marker on the frame
                    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                    cv2.putText(frame, color, (center_x - 20, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return positions

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect color positions
    detect_color_positions(frame, color_ranges)

    # Save frame to the output directory
    output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(output_path, frame)
    frame_count += 1

    # Exit after capturing a certain number of frames (e.g., 100)
    if frame_count >= 100:
        print(f"Captured {frame_count} frames, exiting.")
        break

# Release resources
cap.release()
