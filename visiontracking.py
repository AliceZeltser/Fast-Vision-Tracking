

import cv2
import numpy as np
from picamera2 import Picamera2
import time
from threading import Thread

# Configuration
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
VELOCITY_MULTIPLIER = 4 
MOTION_THRESHOLD = 25   
MIN_CONTOUR_AREA = 50  
SHOW_PREVIEW = True

# Camera Thread
class CameraStream:
    def __init__(self, width, height):
        self.cap = Picamera2()
        config = self.cap.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)},
            controls={"FrameRate": 56.0}
        )
        self.cap.configure(config)
        self.cap.start()
        time.sleep(0.5)  # warm up

        self.frame = None
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True # dont care
        self.thread.start()

    def update(self):
        while not self.stopped:
            self.frame = self.cap.capture_array()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

# Motion detection
def detect_motion(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA]

# Track main object
def track_main_object(prev_gray, gray, contours):
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi_prev = prev_gray[y:y+h, x:x+w]
    roi_now  = gray[y:y+h, x:x+w]

    if roi_prev.size == 0 or roi_now.size == 0:
        return None

    # MAGIC STARTS
    # Returns array of motion vectors (for each pixel)
    flow = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_now, None,
        0.5, 1, 5, 1, 5, 1.1, 0
    )
    # prev image, current image, last motion array (None), recursive scale (pyramind scaling), number of times you scale (orig + x scales), winsize (big = fast but noisy, small = slow), times going through algorithm, pixel neighbourhood (5 or 7 usually), sigma (you are not one) standard deviation (1.1 for poly_n=5), flags
    # MAGIC ENDS

    # get average pixel motion vector
    dx = int(np.mean(flow[...,0]))
    dy = int(np.mean(flow[...,1]))

    # Center of bounding box
    cx = x + w//2
    cy = y + h//2

    # Predicted next point (center + velocity*multiplier)
    px = cx + dx * VELOCITY_MULTIPLIER
    py = cy + dy * VELOCITY_MULTIPLIER

    return {
        "bbox": (x, y, x+w, y+h),
        "center": (cx, cy),
        "predicted": (px, py),
        "velocity": (dx, dy)
    }

# Draw results with FPS
def draw_results(frame, obj, fps):
    if obj:
        x1, y1, x2, y2 = obj["bbox"]
        cx, cy = obj["center"]
        px, py = obj["predicted"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (px, py), 5, (0,0,255), -1)
        cv2.line(frame, (cx, cy), (px, py), (0,0,255), 2)   # Doesn't show up btw

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

def main():
    cam_stream = CameraStream(FRAME_WIDTH, FRAME_HEIGHT)
    while cam_stream.read() is None:
        time.sleep(0.01)  # wait until first frame is ready

    prev_frame = cam_stream.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_time = time.time()
    fps = 0

    print("Starting main object tracker. Press Ctrl+C to stop.")

    try:
        while True:
            frame = cam_stream.read()
            if frame is None:
                continue  # skip if no frame yet

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contours = detect_motion(prev_gray, gray)
            main_obj = track_main_object(prev_gray, gray, contours)  # Actual tracking stuff

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            #

            if SHOW_PREVIEW:
                draw_results(frame, main_obj, fps)
                cv2.imshow("Main Object Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):     # Quit (press 'q')
                    break

            prev_gray = gray.copy()

    except KeyboardInterrupt:
        print("Stopping tracker...")

    cam_stream.stop()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

