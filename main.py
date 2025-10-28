import logging
import cv2
import numpy as np
import json
import socket
import paho.mqtt.client as mqtt
import glob
import os

logging.basicConfig(level=logging.INFO)

# CRS coordinates for conversion (these remain fixed)
BLUE_POS_CRS = (42.33631053975632, -71.09353812118084)
GREEN_POS_CRS = (42.34251643366908, -71.08401409791034)
GUI = True                 # showing the frame 
TEST_MODE = False           # if true, does not burn Colosseum/MQTT
MOVE_THRESHOLD = 0.0001    # Maximum allowed change per frame for the red marker to be considered stable.
STABILITY_FRAMES = 20      # Number of consecutive frames with minimal change before sending an update.


if not TEST_MODE:
    # MQTT setup
    mqtt_broker = "digiran-02.colosseum.net"
    mqtt_port = 1883
    mqtt_topic = "colosseum/update"
    # Use a specific protocol version to avoid ambiguous defaults.
    mqtt_client = mqtt.Client(protocol=mqtt.MQTTv311)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT Broker!")
        else:
            logging.error(f"Failed to connect, return code {rc}")

    mqtt_client.on_connect = on_connect
    # Attempt to connect, but handle DNS/socket errors gracefully so the
    # application can continue running (e.g., in offline or dev environments).
    try:
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()
    except Exception as e:
        # Common root causes include socket.gaierror (DNS) or connection refused.
        logging.error("MQTT connection failed (%s:%s): %s", mqtt_broker, mqtt_port, e)
        logging.info("Continuing without MQTT (mqtt_client set to None).")
        mqtt_client = None

def send_mqtt_message(lat, lon):
    # Ensure lat and lon are Python floats (not np.float32) for JSON serialization.
    message = {
        "id": 2,
        "dynscen_ids": [2],
        "newPosition": {"lat": float(lat), "lng": float(lon)},
        "cmd": "update",
        "type": "MQTTUpdate"
    }
    # If MQTT is disabled/unavailable (None) or we're in TEST_MODE, do not attempt
    # to publish — just log for debugging.
    if TEST_MODE:
        logging.debug("TEST_MODE enabled - not sending MQTT message: %s", message)
        return
    if mqtt_client is None:
        logging.debug("mqtt_client is not connected - cannot send message: %s", message)
        return

    try:
        mqtt_client.publish(mqtt_topic, json.dumps(message))
    except Exception as e:
        logging.error("Failed to publish MQTT message: %s", e)

# Define HSV color ranges for the markers.
# (Adjust these based on your lighting conditions.)
color_ranges = {
    "red":    ((0, 120, 120), (10, 255, 255)),       # Moving marker
    "green":  ((40, 40, 40), (80, 255, 255)),        # Reference marker (top-right)
    "blue":   ((100, 150, 0), (140, 255, 255)),      # Reference marker (top-left)
    "purple": ((130, 50, 50), (160, 255, 255)),      # Reference marker (bottom-left)
    "yellow": ((20, 100, 100), (30, 255, 255))       # Reference marker (bottom-right)
}

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



# ---------------------- Global Calibration Variables ----------------------
calibrated = False           # Flag indicating if calibration is complete.
transformation_matrix = None # Will hold the one-time computed transformation matrix.
blue_cal = None              # Blue marker's rectified position from calibration.
green_cal = None             # Green marker's rectified position from calibration.

# Variables for red marker stability.
stable_count = 0
prev_red_position = None
last_sent_position = None

# Open the video capture device.
def find_camera_device(vendor_id=None, product_id=None, name_substr=None):
    """
    Try to locate a /dev/video* device matching vendor_id/product_id (hex strings
    without 0x) or containing name_substr in the sysfs name entry.
    Returns the device path like '/dev/video5' or None.
    """
    for dev in sorted(glob.glob('/dev/video*')):
        base = os.path.basename(dev)
        name_path = f'/sys/class/video4linux/{base}/name'
        name = ''
        try:
            with open(name_path, 'r') as f:
                name = f.read().strip()
        except Exception:
            name = ''

        if name_substr and name_substr.lower() in name.lower():
            return dev

        # Resolve the device sysfs path and walk up to find idVendor/idProduct
        try:
            devpath = os.path.realpath(f'/sys/class/video4linux/{base}/device')
        except Exception:
            devpath = None

        if devpath:
            cur = devpath
            for _ in range(6):
                vid = os.path.join(cur, 'idVendor')
                pid = os.path.join(cur, 'idProduct')
                if os.path.exists(vid) and os.path.exists(pid):
                    try:
                        with open(vid, 'r') as f:
                            vidv = f.read().strip()
                        with open(pid, 'r') as f:
                            pidv = f.read().strip()
                        # Compare lowercased hex values without leading 0x
                        if vendor_id and product_id and vidv.lower().lstrip('0x') == vendor_id.lower().lstrip('0x') and pidv.lower().lstrip('0x') == product_id.lower().lstrip('0x'):
                            return dev
                    except Exception:
                        pass
                # move up one level
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent

    return None


# Prefer the HDMI USB Camera (lsusb: 32e4:3415) if present, else try indices then nodes.
preferred_dev = find_camera_device(vendor_id='32e4', product_id='3415', name_substr='HDMI USB Camera')
if preferred_dev:
    logging.info('Using detected camera device: %s', preferred_dev)
    cap = cv2.VideoCapture(preferred_dev)
else:
    logging.info('Preferred camera not found; trying indices 0..7')
    cap = None
    for i in range(8):
        test = cv2.VideoCapture(i)
        if test.isOpened():
            logging.info('Opened camera index %d', i)
            cap = test
            break
        test.release()

    if cap is None:
        # try /dev/video* nodes
        for dev in sorted(glob.glob('/dev/video*')):
            try:
                cap_try = cv2.VideoCapture(dev)
                if cap_try.isOpened():
                    logging.info('Opened camera device %s', dev)
                    cap = cap_try
                    break
                cap_try.release()
            except Exception:
                pass

    if cap is None:
        logging.error('No camera could be opened. Falling back to index 0 (may still fail).')
        cap = cv2.VideoCapture(0)
width = 1920
height = 1080
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
                if GUI:
                    cv2.imshow("Cropped & Rectified", cropped_frame)
                    cv2.putText(frame, "Calibrated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    # cv2.imshow("Frame", frame)
            else:
                # Optionally show the original frame while waiting for calibration.
                if GUI:
                    cv2.imshow("Frame", frame)
                    cv2.putText(frame, "Calibrating...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Calibration is done. Use the stored transformation matrix.
            rectified_frame = cv2.warpPerspective(frame, transformation_matrix, (800, 800))
            

            # Now detect only the red marker.
            red_positions = detect_color_positions(frame, {"red": color_ranges["red"]})
            current_red_position = None
            if "red" in red_positions:
                red_pt = np.array([[red_positions["red"]]], dtype="float32")
                red_transformed = cv2.perspectiveTransform(red_pt, transformation_matrix)[0][0]
                # Check boundaries (0 to 800 in rectified frame).
                if (red_transformed[0] < 0 or red_transformed[0] > 800 or 
                    red_transformed[1] < 0 or red_transformed[1] > 800):
                    # logging.info("Red marker is out of bounds. Not updating MQTT.")
                    stable_count = 0
                else:
                    # Compute lat/lon using the stored calibration positions.
                    red_lat, red_lon = rlpos2latlon(blue_cal, green_cal, red_transformed)
                    current_red_position = (red_lat, red_lon)

                    # Stability check.
                    if prev_red_position is not None:
                        if (abs(current_red_position[0] - prev_red_position[0]) < MOVE_THRESHOLD and
                            abs(current_red_position[1] - prev_red_position[1]) < MOVE_THRESHOLD):
                            stable_count += 1
                        else:
                            stable_count = 0
                    else:
                        stable_count = 0

                    prev_red_position = current_red_position

                    if stable_count >= STABILITY_FRAMES:
                        if (last_sent_position is None or
                            abs(current_red_position[0] - last_sent_position[0]) > MOVE_THRESHOLD or
                            abs(current_red_position[1] - last_sent_position[1]) > MOVE_THRESHOLD):
                            if not TEST_MODE:
                                send_mqtt_message(red_lat, red_lon)
                            logging.info(f"Red marker (stable): {red_lat}, {red_lon}")
                            last_sent_position = current_red_position
                            stable_count = 0
            else:
                logging.info("Red marker not found.")
                stable_count = 0

            if GUI:
                # cv2.imshow("Cropped & Rectified", rectified_frame)
                cv2.imshow("Frame", frame)
                if "red" in red_positions:
                    cv2.putText(frame, f"Red: {current_red_position}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif prev_red_position is not None:
                    cv2.putText(frame, f"Red: {prev_red_position}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error("An error occurred: %s", e)
finally:
    cap.release()
    if GUI:
        cv2.destroyAllWindows()
