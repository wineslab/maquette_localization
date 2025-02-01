import logging
import cv2
import numpy as np
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO)

BLUE_POS_CRS =  (42.33631053975632, -71.09353812118084)         # correspoding lat and lon to blue color
GREEN_POS_CRS = (42.34251643366908, -71.08401409791034)         # correspoding lat and lon to green color
GUI = True

# MQTT setup
mqtt_broker = "digiran-02.colosseum.net"
mqtt_port = 1883
mqtt_topic = "colosseum/update"
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to MQTT Broker!")
    else:
        logging.error(f"Failed to connect, return code {rc}")

mqtt_client.on_connect = on_connect
mqtt_client.connect(mqtt_broker, mqtt_port, 60)
mqtt_client.loop_start()

def send_mqtt_message(lat, lon):
    message = {
        "id": 0,
        "dynscen_ids": [0],
        "newPosition": {"lat": lat, "lng": lon},
        "cmd": "update",
        "type": "MQTTUpdate"
    }
    mqtt_client.publish(mqtt_topic, str(message))

# Define color ranges in HSV (tweak these values based on your markers)
color_ranges = {
    "red": ((0, 120, 70), (10, 255, 255)),  # Lower and upper HSV bounds for red
    "green": ((40, 40, 40), (80, 255, 255)),  # Lower and upper HSV bounds for green
    "blue": ((100, 150, 0), (140, 255, 255)),  # Lower and upper HSV bounds for blue
}

def detect_color_positions(frame, color_ranges):
    """
    Detects the positions of specified colors in a given frame.

    Args:
        frame (numpy.ndarray): The input image frame in BGR format.
        color_ranges (dict): A dictionary where keys are color names (str) and values are tuples containing
                             the lower and upper HSV bounds for the color (list of int).

    Returns:
        dict: A dictionary where keys are color names (str) and values are tuples containing the (x, y) positions
              of the detected color centers in the frame.
    """
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


def rlpos2latlon(blue_pos, green_pos, red_pos):
    """
    Calculate the relative position of the red marker based on the positions of the blue and green markers.

    Args:
        blue_pos (tuple): The (x, y) position of the blue marker in the frame.
        green_pos (tuple): The (x, y) position of the green marker in the frame.
        red_pos (tuple): The (x, y) position of the red marker in the frame.

    Returns:
        tuple: The (latitude, longitude) of the red marker.
    """
    # Calculate the relative distances in the frame
    blue_to_green_x = green_pos[0] - blue_pos[0]
    blue_to_green_y = green_pos[1] - blue_pos[1]
    blue_to_red_x = red_pos[0] - blue_pos[0]
    blue_to_red_y = red_pos[1] - blue_pos[1]

    # Calculate the relative position in CRS
    blue_to_green_lat = GREEN_POS_CRS[0] - BLUE_POS_CRS[0]
    blue_to_green_lon = GREEN_POS_CRS[1] - BLUE_POS_CRS[1]

    red_lat = BLUE_POS_CRS[0] + (blue_to_red_y / blue_to_green_y) * blue_to_green_lat if blue_to_green_y != 0 else BLUE_POS_CRS[0]
    red_lon = BLUE_POS_CRS[1] + (blue_to_red_x / blue_to_green_x) * blue_to_green_lon if blue_to_green_x != 0 else BLUE_POS_CRS[1]

    return red_lat, red_lon


cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect color positions
        positions = detect_color_positions(frame, color_ranges)
        if "blue" in positions and "green" in positions and "red" in positions:
            red_lat, red_lon = rlpos2latlon(positions["blue"], positions["green"], positions["red"])
            location_data = {"latitude": red_lat, "longitude": red_lon}
            send_mqtt_message(red_lat, red_lon)
            logging.info(f"Red marker coordinates: Latitude = {red_lat}, Longitude = {red_lon}")

        # Show the frame in a window if GUI is enabled
        if GUI:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    if GUI:
        cv2.destroyAllWindows()
