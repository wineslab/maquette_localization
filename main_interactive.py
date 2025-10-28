import logging
import cv2
import numpy as np
import json
import paho.mqtt.client as mqtt
import glob
import os

logging.basicConfig(level=logging.INFO)

# CRS coordinates for conversion (these remain fixed)
BLUE_POS_CRS = (42.33631053975632, -71.09353812118084)
GREEN_POS_CRS = (42.34251643366908, -71.08401409791034)

# Configuration
GUI = True
TEST_MODE = False
MOVE_THRESHOLD = 0.0001
STABILITY_FRAMES = 20
DISPLAY_SCALE = 0.5  # Scale factor for display windows (0.5 = 50% size)

# MQTT setup
if not TEST_MODE:
    mqtt_broker = "digiran-02.colosseum.net"
    mqtt_port = 1883
    mqtt_topic = "colosseum/update"
    mqtt_client = mqtt.Client(protocol=mqtt.MQTTv311)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT Broker!")
        else:
            logging.error(f"Failed to connect, return code {rc}")

    mqtt_client.on_connect = on_connect
    try:
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()
    except Exception as e:
        logging.error("MQTT connection failed (%s:%s): %s", mqtt_broker, mqtt_port, e)
        logging.info("Continuing without MQTT (mqtt_client set to None).")
        mqtt_client = None


def send_mqtt_message(node_id, lat, lon):
    """Send MQTT message for a node update."""
    message = {
        "id": node_id,
        "dynscen_ids": [node_id],
        "newPosition": {"lat": float(lat), "lng": float(lon)},
        "cmd": "update",
        "type": "MQTTUpdate"
    }
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


# Define HSV color ranges for all markers (now all are trackable nodes)
color_ranges = {
    "red":    ((0, 120, 120), (10, 255, 255)),
    "green":  ((40, 40, 40), (80, 255, 255)),
    "blue":   ((100, 150, 0), (140, 255, 255)),
    "purple": ((130, 50, 50), (160, 255, 255)),
    "yellow": ((20, 100, 100), (30, 255, 255))
}

# Map colors to node IDs for MQTT
color_to_node_id = {
    "red": 2,
    "green": 3,
    "blue": 4,
    "purple": 5,
    "yellow": 6
}


def detect_color_positions(frame, ranges):
    """
    Detects markers in the frame and returns a dictionary mapping color names to (x, y) positions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    positions = {}
    for color, (lower, upper) in ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 20:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    positions[color] = (center_x, center_y)
    return positions


def perspective_transform(image, src_points, output_size=(800, 800)):
    """
    Applies a perspective transform to crop and rectify the region defined by src_points.
    src_points: a 4x2 array of corner positions (top-left, top-right, bottom-right, bottom-left).
    Returns the rectified image and the transformation matrix.
    """
    width, height = output_size
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, matrix, (width, height))
    return result, matrix


def pixel_to_latlon(pixel_pos, output_size=(800, 800)):
    """
    Converts a pixel position in the rectified frame to lat/lon coordinates.
    Uses linear interpolation based on the reference CRS coordinates.
    """
    width, height = output_size
    x, y = pixel_pos
    
    # Normalize to [0, 1] range
    x_norm = x / width
    y_norm = y / height
    
    # Linear interpolation between the reference points
    # Assuming BLUE is top-left and GREEN is bottom-right in CRS space
    lat = BLUE_POS_CRS[0] + y_norm * (GREEN_POS_CRS[0] - BLUE_POS_CRS[0])
    lon = BLUE_POS_CRS[1] + x_norm * (GREEN_POS_CRS[1] - BLUE_POS_CRS[1])
    
    return lat, lon


class InteractiveCalibrator:
    """Handles interactive corner selection for calibration."""
    
    def __init__(self, window_name="Calibration"):
        self.window_name = window_name
        self.corners = []
        self.current_frame = None
        self.display_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for corner selection."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            # Scale coordinates back to original frame size
            actual_x = int(x / DISPLAY_SCALE)
            actual_y = int(y / DISPLAY_SCALE)
            self.corners.append((actual_x, actual_y))
            logging.info(f"Corner {len(self.corners)} selected at ({actual_x}, {actual_y})")
            self.update_display()
    
    def update_display(self):
        """Update the display with current corner selections."""
        if self.current_frame is None:
            return
        
        self.display_frame = self.current_frame.copy()
        
        # Draw selected corners
        for i, corner in enumerate(self.corners):
            cv2.circle(self.display_frame, corner, 10, (0, 255, 0), -1)
            cv2.putText(self.display_frame, f"{i+1}", 
                       (corner[0] + 15, corner[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw lines between corners
        if len(self.corners) > 1:
            for i in range(len(self.corners)):
                cv2.line(self.display_frame, self.corners[i], 
                        self.corners[(i + 1) % len(self.corners)], 
                        (0, 255, 0), 2)
        
        # Instructions
        instruction = f"Click corner {len(self.corners) + 1}/4"
        if len(self.corners) == 4:
            instruction = "Press ENTER to confirm, 'r' to reset"
        
        cv2.putText(self.display_frame, instruction, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(self.display_frame, "Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Scale down for display if needed
        if DISPLAY_SCALE != 1.0:
            height, width = self.display_frame.shape[:2]
            new_width = int(width * DISPLAY_SCALE)
            new_height = int(height * DISPLAY_SCALE)
            display_scaled = cv2.resize(self.display_frame, (new_width, new_height))
            cv2.imshow(self.window_name, display_scaled)
        else:
            cv2.imshow(self.window_name, self.display_frame)
    
    def calibrate(self, cap):
        """
        Interactive calibration process.
        Returns the transformation matrix or None if cancelled.
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        logging.info("Starting interactive calibration...")
        logging.info("Click 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame during calibration")
                return None
            
            self.current_frame = frame
            if len(self.corners) == 0:
                self.update_display()
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                # Reset corners
                self.corners = []
                logging.info("Corners reset")
                self.update_display()
            elif key == 13 and len(self.corners) == 4:  # Enter key
                # Confirm calibration
                src_points = np.float32(self.corners)
                _, matrix = perspective_transform(frame, src_points)
                logging.info("Calibration complete!")
                cv2.destroyWindow(self.window_name)
                return matrix
            elif key == ord('q'):
                # Cancel
                logging.info("Calibration cancelled")
                cv2.destroyWindow(self.window_name)
                return None
        
        return None


def find_camera_device(vendor_id=None, product_id=None, name_substr=None):
    """
    Try to locate a /dev/video* device matching vendor_id/product_id or name_substr.
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
                        if vendor_id and product_id and vidv.lower().lstrip('0x') == vendor_id.lower().lstrip('0x') and pidv.lower().lstrip('0x') == product_id.lower().lstrip('0x'):
                            return dev
                    except Exception:
                        pass
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent

    return None


class NodeTracker:
    """Tracks multiple colored nodes and manages stability detection."""
    
    def __init__(self, colors, move_threshold=0.0001, stability_frames=20):
        self.colors = colors
        self.move_threshold = move_threshold
        self.stability_frames = stability_frames
        
        # State for each color
        self.stable_counts = {color: 0 for color in colors}
        self.prev_positions = {color: None for color in colors}
        self.last_sent_positions = {color: None for color in colors}
    
    def update(self, color, current_position):
        """
        Update tracking for a specific color node.
        Returns True if a stable position should be sent, False otherwise.
        """
        if self.prev_positions[color] is not None:
            lat_diff = abs(current_position[0] - self.prev_positions[color][0])
            lon_diff = abs(current_position[1] - self.prev_positions[color][1])
            
            if lat_diff < self.move_threshold and lon_diff < self.move_threshold:
                self.stable_counts[color] += 1
            else:
                self.stable_counts[color] = 0
        else:
            self.stable_counts[color] = 0
        
        self.prev_positions[color] = current_position
        
        # Check if stable enough to send
        if self.stable_counts[color] >= self.stability_frames:
            if self.last_sent_positions[color] is None:
                # First time sending
                self.last_sent_positions[color] = current_position
                self.stable_counts[color] = 0
                return True
            else:
                # Check if position changed enough since last send
                lat_diff = abs(current_position[0] - self.last_sent_positions[color][0])
                lon_diff = abs(current_position[1] - self.last_sent_positions[color][1])
                
                if lat_diff > self.move_threshold or lon_diff > self.move_threshold:
                    self.last_sent_positions[color] = current_position
                    self.stable_counts[color] = 0
                    return True
        
        return False
    
    def reset(self, color):
        """Reset tracking state for a specific color."""
        self.stable_counts[color] = 0


def main():
    # Camera setup
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
            logging.error('No camera could be opened. Falling back to index 0.')
            cap = cv2.VideoCapture(0)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Interactive calibration
    calibrator = InteractiveCalibrator()
    transformation_matrix = calibrator.calibrate(cap)
    
    if transformation_matrix is None:
        logging.error("Calibration failed or cancelled")
        cap.release()
        return

    # Initialize node tracker
    tracker = NodeTracker(list(color_ranges.keys()), MOVE_THRESHOLD, STABILITY_FRAMES)

    # Main tracking loop
    try:
        logging.info("Starting node tracking...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break

            # Apply perspective transform
            rectified_frame = cv2.warpPerspective(frame, transformation_matrix, (800, 800))
            
            # Detect all color markers
            positions = detect_color_positions(frame, color_ranges)
            
            # Process each detected marker
            for color, pixel_pos in positions.items():
                # Transform to rectified coordinates
                pt = np.array([[pixel_pos]], dtype="float32")
                transformed = cv2.perspectiveTransform(pt, transformation_matrix)[0][0]
                
                # Check if within bounds
                if (transformed[0] < 0 or transformed[0] > 800 or 
                    transformed[1] < 0 or transformed[1] > 800):
                    tracker.reset(color)
                    continue
                
                # Convert to lat/lon
                lat, lon = pixel_to_latlon(transformed)
                
                # Update tracker and send if stable
                should_send = tracker.update(color, (lat, lon))
                if should_send:
                    node_id = color_to_node_id[color]
                    if not TEST_MODE:
                        send_mqtt_message(node_id, lat, lon)
                    logging.info(f"{color.upper()} node (ID {node_id}): {lat:.8f}, {lon:.8f}")
                
                # Visualization
                if GUI:
                    cv2.circle(frame, pixel_pos, 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"{color}", 
                               (pixel_pos[0] - 20, pixel_pos[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frames
            if GUI:
                # Scale down frames for display
                if DISPLAY_SCALE != 1.0:
                    h_frame, w_frame = frame.shape[:2]
                    frame_display = cv2.resize(frame, (int(w_frame * DISPLAY_SCALE), int(h_frame * DISPLAY_SCALE)))
                    h_rect, w_rect = rectified_frame.shape[:2]
                    rectified_display = cv2.resize(rectified_frame, (int(w_rect * DISPLAY_SCALE), int(h_rect * DISPLAY_SCALE)))
                    cv2.imshow("Original", frame_display)
                    cv2.imshow("Rectified Map", rectified_display)
                else:
                    cv2.imshow("Original", frame)
                    cv2.imshow("Rectified Map", rectified_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error("An error occurred: %s", e)
    finally:
        cap.release()
        if GUI:
            cv2.destroyAllWindows()
        if not TEST_MODE and mqtt_client:
            mqtt_client.loop_stop()


if __name__ == "__main__":
    main()
