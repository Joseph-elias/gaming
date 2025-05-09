import cv2
import mediapipe as mp
import pyautogui
from math import hypot
import time

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Store previous finger positions to calculate movement
prev_index_finger = None
prev_middle_finger = None
prev_time = None

# Sensitivity factor (higher value = more sensitive)
SENSITIVITY = 1.5  # You can experiment with different values

# Thresholds for gesture detection
SWIPE_DISTANCE_THRESHOLD = 60  # Minimum pixel movement for swipe
SWIPE_VELOCITY_THRESHOLD = 0.7  # Minimum velocity (pixels/ms) for swipe
SWIPE_COOLDOWN = 0.5  # Seconds between swipes
CLICK_THRESHOLD = 20  # Distance for "click" gesture

last_swipe_time = 0

# Get the screen size
screen_width, screen_height = pyautogui.size()

def simulate_swipe(start, end):
    # Calculate the movement of the finger
    dx = (end[0] - start[0]) * SENSITIVITY  # Apply sensitivity to horizontal movement
    dy = (end[1] - start[1]) * SENSITIVITY  # Apply sensitivity to vertical movement

    # Simulate a swipe from start to end coordinates using PyAutoGUI
    pyautogui.moveTo(start[0], start[1])  # Move the mouse to the starting point
    pyautogui.dragTo(start[0] + dx, start[1] + dy, duration=0.3)  # Drag the mouse to the end point

def simulate_click(start):
    # Simulate a click (press and release)
    pyautogui.moveTo(start[0], start[1])  # Move the mouse to the click position
    pyautogui.mouseDown()  # Press the mouse button
    pyautogui.mouseUp()    # Release the mouse button

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a natural selfie view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as required by MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index, middle, and thumb tips landmarks
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert the landmarks to pixel coordinates
            height, width, _ = frame.shape
            index_finger_coords = (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
            middle_finger_coords = (int(middle_finger_tip.x * width), int(middle_finger_tip.y * height))
            thumb_coords = (int(thumb_tip.x * width), int(thumb_tip.y * height))

            # Map the index finger coordinates to the screen size
            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)
            pyautogui.moveTo(screen_x, screen_y)

            # Draw the finger landmarks on the frame
            cv2.circle(frame, index_finger_coords, 10, (0, 0, 255), -1)
            cv2.circle(frame, middle_finger_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, thumb_coords, 10, (255, 0, 0), -1)

            # Calculate the distance between the index and middle finger tips
            dist_index_middle = hypot(index_finger_coords[0] - middle_finger_coords[0], 
                                      index_finger_coords[1] - middle_finger_coords[1])

            # Calculate the distance between the thumb and index finger
            dist_thumb_index = hypot(thumb_coords[0] - index_finger_coords[0], 
                                      thumb_coords[1] - index_finger_coords[1])

            # Detect hold gesture (when thumb and index finger are close together)
            if dist_thumb_index < CLICK_THRESHOLD:
                # Hold the mouse button
                pyautogui.mouseDown()
                print("Hold detected!")
            else:
                # Release the mouse button when fingers move apart
                pyautogui.mouseUp()
                print("Release detected!")

            # Calculate swipe movement and velocity
            if prev_index_finger is not None and prev_time is not None:
                dx = index_finger_coords[0] - prev_index_finger[0]
                dy = index_finger_coords[1] - prev_index_finger[1]
                dist = hypot(dx, dy)
                dt = current_time - prev_time
                velocity = dist / (dt * 1000) if dt > 0 else 0  # pixels/ms

                # Detect swipe gesture: fingers apart, fast movement, and cooldown
                if (dist_index_middle > SWIPE_DISTANCE_THRESHOLD and
                    dist > SWIPE_DISTANCE_THRESHOLD and
                    velocity > SWIPE_VELOCITY_THRESHOLD and
                    (current_time - last_swipe_time) > SWIPE_COOLDOWN):
                    simulate_swipe(prev_index_finger, index_finger_coords)
                    print(f"Swipe detected! dx={dx}, dy={dy}, velocity={velocity:.2f} px/ms")
                    last_swipe_time = current_time

            # Update the previous finger positions and time for the next frame
            prev_index_finger = index_finger_coords
            prev_middle_finger = middle_finger_coords
            prev_time = current_time

    # Display the frame with hand landmarks
    cv2.imshow("Hand Tracking with Click and Swipe Detection", frame)

    # Check if 'q' key is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
