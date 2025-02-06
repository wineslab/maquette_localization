**Overview**  
This script captures video from a camera, detects colored markers using OpenCV, and applies a perspective transform to rectify the view. It uses four reference markers (blue, green, purple, yellow) for calibration. Once calibrated, it tracks a red marker, computes its geographic coordinates (lat/lon) via linear interpolation, and publishes these coordinates to an MQTT broker when the marker remains stable over several frames.

**Key Components**  
- **Marker Detection:** Uses HSV color thresholds to locate markers in each frame.
- **Calibration & Transformation:** Detects four fixed markers to compute a perspective transformation for a rectified (800×800) view.
- **Position Calculation:** Converts the red marker’s rectified position into latitude/longitude using known blue and green reference CRS coordinates.
- **MQTT Communication:** Publishes position updates via MQTT when the red marker’s movement is minimal for a set number of frames.
- **GUI Option:** Displays both the original and rectified frames for visual debugging (if enabled).

**Running the Code**  
- **Local Execution:** Ensure you have Python with the required libraries (`opencv-python`, `numpy`, `paho-mqtt`) installed.
- **Docker Deployment:** It’s recommended to use Docker Compose for easy deployment. Use the following command to bring up the container:
  ```bash
  docker-compose up
  ```