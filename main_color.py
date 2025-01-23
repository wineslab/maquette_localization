import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define color ranges in HSV (tweak these values based on your markers)
color_ranges = {
    "red": ((0, 120, 70), (10, 255, 255)),  # Lower and upper HSV bounds for red
    "green": ((40, 40, 40), (80, 255, 255)),  # Lower and upper HSV bounds for green
    "blue": ((100, 150, 0), (140, 255, 255)),  # Lower and upper HSV bounds for blue
}

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect color positions
    color_positions = detect_color_positions(frame, color_ranges)

    # Display relative positions
    if len(color_positions) > 1:
        colors = list(color_positions.keys())
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                color1, color2 = colors[i], colors[j]
                x1, y1 = color_positions[color1]
                x2, y2 = color_positions[color2]
                rel_x, rel_y = x2 - x1, y2 - y1
                print(f"{color2} relative to {color1}: ({rel_x}, {rel_y})")

    # Display the frame
    cv2.imshow("Color-Based Localization", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
    