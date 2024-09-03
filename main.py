import cv2
import time
from datetime import datetime

# Sensitivity Parameters
CONTOUR_AREA_THRESHOLD = 4000  # Minimum area in pixels to be considered as motion
MOTION_RESET_TIME = 3  # Time in seconds after which recording stops if no motion is detected

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Variables for motion detection
motion_detected = False
last_motion_time = time.time()
recording = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply the background subtractor
    fg_mask = bg_subtractor.apply(frame)

    # Use erosion and dilation to remove noise
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours to detect motion
    contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = any(cv2.contourArea(contour) > CONTOUR_AREA_THRESHOLD for contour in contours)

    if motion:
        last_motion_time = time.time()
        if not recording:
            # Start recording with a formatted timestamp
            timestamp = datetime.now().strftime("%m_%d_%H-%M-%S")
            output_filename = f'recording_{timestamp}.avi'
            out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
        print(f"Motion detected! Recording to {output_filename}...")
    else:
        if recording and time.time() - last_motion_time >= MOTION_RESET_TIME:
            # Stop recording after the defined time period of no motion
            print("No motion detected. Stopping recording...")
            recording = False
            if out:
                out.release()
                out = None

    if recording and out:
        out.write(frame)

    # Display the video feed with the foreground mask
    cv2.imshow("Motion Detection", fg_mask)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
