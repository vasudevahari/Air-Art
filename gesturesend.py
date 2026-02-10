# Author: @suryaprabhaz
import cv2
import pyperclip
import requests
import time
import sys
import numpy as np
import math

# Custom Hand Tracker using OpenCV (Fallback for MediaPipe)
class SimpleHandTracker:
    def __init__(self):
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        self.calibrated = False
    
    def calibrate(self, frame):
        # Sample center of frame for skin color
        h, w, _ = frame.shape
        center_region = frame[h//2-20:h//2+20, w//2-20:w//2+20]
        hsv_roi = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        
        # Calculate mean and std dev
        mean = np.mean(hsv_roi, axis=(0,1))
        std = np.std(hsv_roi, axis=(0,1))
        
        # Set dynamic thresholds (mean +/- 3*std, clamped)
        self.lower_skin = np.array([max(0, mean[0]-30), max(0, mean[1]-40), max(0, mean[2]-40)], dtype=np.uint8)
        self.upper_skin = np.array([min(180, mean[0]+30), 255, 255], dtype=np.uint8)
        self.calibrated = True
        print(f"[⚙️] Calibrated Skin Color: Lower={self.lower_skin}, Upper={self.upper_skin}")

    def find_hands(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5,5), 100)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 3000: # Hand must be big enough
                return max_contour
        return None

    def check_fist(self, contour, frame, show_debug=False):
        if contour is None:
            return False, frame
            
        # Draw only simple outline by default
        if show_debug:
             cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
             hull = cv2.convexHull(contour)
             cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)

        hull_indices = cv2.convexHull(contour, returnPoints=False)
        
        if len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                count_defects = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        
                        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                        s = (a+b+c)/2
                        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
                        d = (2*ar)/a
                        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                        
                        if angle <= 90 and d > 30:
                            count_defects += 1
                
                # FIST = Low defects (0, 1, or 2)
                if count_defects <= 1: 
                    return True, frame 
                    
            except Exception:
                pass 
        return False, frame

# State
last_trigger_time = 0
cooldown_seconds = 3
SERVER_URL = "http://127.0.0.1:5000/save"
fist_frames = 0 # Debouncing counter

# Initialize Tracker
tracker = SimpleHandTracker()
USE_CV_FALLBACK = True
SHOW_DEBUG_LINES = False # Cleaner UI by default

def send_text_to_server(text):
    try:
        res = requests.post(SERVER_URL, json={"text": text}, timeout=2)
        if res.status_code == 200:
            print(f"\n[✅] SUCCESS: Sent to phone")
            return True
    except:
        return False
    return False

print("\n" + "="*50)
print(f"🖐  GESTURE SENDER ACTIVATED (OpenCV Mode)")
print("="*50)
print("1. [IMPORTANT] Place your hand in the center box")
print("   and press 'C' to calibrate for your lighting.")
print("2. Copy some text.")
print("3. Show a FIST ✊ to send.")
print("4. Press 'D' to toggle debug lines.")
print("="*50 + "\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    frame = cv2.flip(frame, 1)
    status_text = "Waiting..."
    color = (255, 255, 0)
    trigger_action = False

    # Calibration UI
    if not tracker.calibrated:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (w//2-20, h//2-20), (w//2+20, h//2+20), (0, 0, 255), 2)
        cv2.putText(frame, "Place hand in box & press 'C'", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Track Hand
    if USE_CV_FALLBACK:
        contour = tracker.find_hands(frame)
        is_fist, frame = tracker.check_fist(contour, frame, SHOW_DEBUG_LINES)
        
        if is_fist:
            fist_frames += 1
        else:
            fist_frames = 0
            
        # Require 10 consecutive frames of FIST to trigger (Debouncing)
        if fist_frames > 10:
            status_text = "✊ Fist Detected!"
            color = (0, 255, 0)
            trigger_action = True

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break
    elif key == ord('f'):
        trigger_action = True
    elif key == ord('c'):
        tracker.calibrate(frame)
    elif key == ord('d'):
        SHOW_DEBUG_LINES = not SHOW_DEBUG_LINES

    if trigger_action:
        current_time = time.time()
        if current_time - last_trigger_time > cooldown_seconds:
            print("[✊] Trigger detected!")
            text = pyperclip.paste()
            if text and text.strip():
                if send_text_to_server(text):
                    last_trigger_time = current_time
                    fist_frames = 0 # Reset
                    cv2.putText(frame, "SENT!", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    cv2.imshow("Gesture Drop - Sender", frame)
                    cv2.waitKey(500) # Show sent message for 0.5s

    # UI Overlay
    cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Gesture Drop - Sender", frame)

cap.release()
cv2.destroyAllWindows()