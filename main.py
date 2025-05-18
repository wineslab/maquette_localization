import logging
import cv2
import numpy as np
import json
import paho.mqtt.client as mqtt
import yaml
import os

logging.basicConfig(level=logging.INFO)

# Load configuration from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Extract config values
gui = config.get('gui', True)
test_mode = config.get('test_mode', False)
move_threshold = config.get('move_threshold', 0.0001)
stability_frames = config.get('stability_frames', 20)

mqtt_config = config['mqtt']
mqtt_broker = mqtt_config['broker']
mqtt_port = mqtt_config['port']
mqtt_topic = mqtt_config['topic']
mqtt_id = mqtt_config['id']
mqtt_dynscen_ids = mqtt_config['dynscen_ids']

webcam_config = config['webcam']
webcam_index = webcam_config.get('index', 0)
width = webcam_config.get('width', 1920)
height = webcam_config.get('height', 1080)

crs = config['crs']
BLUE_POS_CRS = tuple(crs['blue'])
GREEN_POS_CRS = tuple(crs['green'])

color_ranges = {}
for color, val in config['color_ranges'].items():
    color_ranges[color] = (tuple(val['lower']), tuple(val['upper']))

# ---------------------- Global Calibration Variables ----------------------
calibrated = False           # Flag indicating if calibration is complete.
transformation_matrix = None # Will hold the one-time computed transformation matrix.
blue_cal = None              # Blue marker's rectified position from calibration.
green_cal = None             # Green marker's rectified position from calibration.

# Variables for moving node stability (dicts for each node)
moving_nodes = config.get('moving_nodes', ['red'])
stable_count = {node: 0 for node in moving_nodes}
prev_position = {node: None for node in moving_nodes}
last_sent_position = {node: None for node in moving_nodes}

if not test_mode:
    # MQTT setup
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
    # Ensure lat and lon are Python floats (not np.float32) for JSON serialization.
    message = {
        "id": mqtt_id,
        "dynscen_ids": mqtt_dynscen_ids,
        "newPosition": {"lat": float(lat), "lng": float(lon)},
        "cmd": "update",
        "type": "MQTTUpdate"
    }
    mqtt_client.publish(mqtt_topic, json.dumps(message))

# Define HSV color ranges for the markers.
# (Adjust these based on your lighting conditions.)
# color_ranges = {
#     "red":    ((0, 120, 120), (10, 255, 255)),       # Moving marker
#     "green":  ((40, 40, 40), (80, 255, 255)),        # Reference marker (top-right)
#     "blue":   ((100, 150, 0), (140, 255, 255)),      # Reference marker (top-left)
#     "purple": ((130, 50, 50), (160, 255, 255)),      # Reference marker (bottom-left)
#     "yellow": ((20, 100, 100), (30, 255, 255))       # Reference marker (bottom-right)
# }

def detect_color_positions(frame, ranges):
    """
    Detects markers in the frame and returns a dictionary mapping color names to (x, y) positions.
    The 'ranges' parameter is a dictionary containing only the desired colors.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    positions = {}
    for color, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the largest contour (assumed to be the marker)
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 20:  # Filter out noise
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    positions[color] = (center_x, center_y)
                    # Draw for visualization.
                    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                    cv2.putText(frame, color, (center_x - 20, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return positions

def perspective_transform(image, src_points):
    """
    Applies a perspective transform to crop and rectify the region defined by src_points.
    src_points: a 4x2 array of marker positions (in the original frame).
    Returns the rectified image and the transformation matrix.
    """
    width = 800   # Desired output width.
    height = 800  # Desired output height.
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (width, height))
    return result, matrix

def rlpos2latlon(blue_pos, green_pos, red_pos):
    """
    Computes the red marker's latitude and longitude using linear interpolation
    based on the positions of the blue and green markers (in the rectified frame).
    """
    blue_to_green_x = green_pos[0] - blue_pos[0]
    blue_to_green_y = green_pos[1] - blue_pos[1]
    blue_to_red_x = red_pos[0] - blue_pos[0]
    blue_to_red_y = red_pos[1] - blue_pos[1]

    blue_to_green_lat = GREEN_POS_CRS[0] - BLUE_POS_CRS[0]
    blue_to_green_lon = GREEN_POS_CRS[1] - BLUE_POS_CRS[1]

    if blue_to_green_x == 0 or blue_to_green_y == 0:
        return BLUE_POS_CRS

    red_lat = BLUE_POS_CRS[0] + (blue_to_red_y / blue_to_green_y) * blue_to_green_lat
    red_lon = BLUE_POS_CRS[1] + (blue_to_red_x / blue_to_green_x) * blue_to_green_lon
    return red_lat, red_lon

# Open the video capture device.
cap = cv2.VideoCapture(webcam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if not calibrated:
            # In calibration mode, detect all reference markers.
            full_ranges = {
                "blue": color_ranges["blue"],
                "green": color_ranges["green"],
                "purple": color_ranges["purple"],
                "yellow": color_ranges["yellow"],
                "red": color_ranges["red"]  # You can also display red during calibration.
            }
            positions = detect_color_positions(frame, full_ranges)
            # Check if all four reference markers are detected.
            if all(marker in positions for marker in ["blue", "green", "purple", "yellow"]):
                # Use an ordering that matches your physical layout.
                src_points = np.float32([
                    positions["purple"],  # top-left
                    positions["blue"],    # top-right
                    positions["yellow"],  # bottom-right
                    positions["green"]    # bottom-left
                ])
                # Compute the transformation matrix and get the rectified image.
                cropped_frame, transformation_matrix = perspective_transform(frame, src_points)
                # Transform the blue and green marker positions once.
                blue_pt = np.array([[positions["blue"]]], dtype="float32")
                green_pt = np.array([[positions["green"]]], dtype="float32")
                blue_cal = cv2.perspectiveTransform(blue_pt, transformation_matrix)[0][0]
                green_cal = cv2.perspectiveTransform(green_pt, transformation_matrix)[0][0]
                calibrated = True
                logging.info("Calibrated.")
                if gui:
                    cv2.imshow("Cropped & Rectified", cropped_frame)
                    cv2.putText(frame, "Calibrated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    # cv2.imshow("Frame", frame)
            else:
                # Optionally show the original frame while waiting for calibration.
                if gui:
                    cv2.imshow("Frame", frame)
                    cv2.putText(frame, "Calibrating...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Calibration is done. Use the stored transformation matrix.
            rectified_frame = cv2.warpPerspective(frame, transformation_matrix, (800, 800))

            # Detect all moving nodes
            moving_ranges = {node: color_ranges[node] for node in moving_nodes}
            node_positions = detect_color_positions(frame, moving_ranges)
            current_positions = {}
            for node in moving_nodes:
                if node in node_positions:
                    node_pt = np.array([[node_positions[node]]], dtype="float32")
                    node_transformed = cv2.perspectiveTransform(node_pt, transformation_matrix)[0][0]
                    # Check boundaries (0 to 800 in rectified frame).
                    if (node_transformed[0] < 0 or node_transformed[0] > 800 or 
                        node_transformed[1] < 0 or node_transformed[1] > 800):
                        stable_count[node] = 0
                    else:
                        # Compute lat/lon using the stored calibration positions.
                        node_lat, node_lon = rlpos2latlon(blue_cal, green_cal, node_transformed)
                        current_positions[node] = (node_lat, node_lon)

                        # Stability check.
                        if prev_position[node] is not None:
                            if (abs(current_positions[node][0] - prev_position[node][0]) < move_threshold and
                                abs(current_positions[node][1] - prev_position[node][1]) < move_threshold):
                                stable_count[node] += 1
                            else:
                                stable_count[node] = 0
                        else:
                            stable_count[node] = 0

                        prev_position[node] = current_positions[node]

                        if stable_count[node] >= stability_frames:
                            if (last_sent_position[node] is None or
                                abs(current_positions[node][0] - last_sent_position[node][0]) > move_threshold or
                                abs(current_positions[node][1] - last_sent_position[node][1]) > move_threshold):
                                if not test_mode:
                                    send_mqtt_message(node_lat, node_lon)
                                logging.info(f"{node.capitalize()} marker (stable): {node_lat}, {node_lon}")
                                last_sent_position[node] = current_positions[node]
                                stable_count[node] = 0
                else:
                    stable_count[node] = 0
                    logging.info(f"{node.capitalize()} marker not found.")

            if gui:
                cv2.imshow("Frame", frame)
                for node in moving_nodes:
                    pos = current_positions.get(node) or prev_position.get(node)
                    if pos is not None:
                        cv2.putText(frame, f"{node.capitalize()}: {pos}", (50, 100 + 30 * moving_nodes.index(node)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error("An error occurred: %s", e)
finally:
    cap.release()
    if gui:
        cv2.destroyAllWindows()
