import cv2
import mediapipe as mp
import pyautogui
from time import time
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   min_detection_confidence=0.6,
                                   min_tracking_confidence=0.6)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Landmark indices
NOSE_TIP = 1
CHIN = 152
LEFT_EYE = 33
RIGHT_EYE = 263
FOREHEAD = 10
EYE_TOP = 159
EYE_BOTTOM = 145

# Calibration
CALIBRATION_DURATION = 5
calibration_start = time()
calibrated = False
calibration_data = []

neutral_values = {}
TOLERANCE = {
    "left_right": 25,
    "up": 15,
    "down": 25
}

# Gesture state
last_action_time = time()
ACTION_COOLDOWN = 0.8
current_direction = None

# Eye blink to trigger skate
EYE_CLOSED_THRESHOLD = 3.0  # pixels
eye_closed_triggered = False

def trigger_action(action):
    global last_action_time, current_direction
    now = time()
    if now - last_action_time >= ACTION_COOLDOWN or action != current_direction:
        pyautogui.press(action)
        print(f"🔹 Action: {action}")
        current_direction = action
        last_action_time = now

def draw_bar(image, label, value, neutral, tol, x, y, color):
    bar_length = 200
    diff = value - neutral
    pct = np.clip((diff + tol) / (2 * tol), 0, 1)
    fill = int(pct * bar_length)
    cv2.rectangle(image, (x, y), (x + bar_length, y + 20), (80, 80, 80), 2)
    cv2.rectangle(image, (x, y), (x + fill, y + 20), color, -1)
    cv2.putText(image, f"{label}: {int(value)}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    height, width, _ = frame.shape
    direction_label = "Neutral"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Get landmark positions
        nose = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        forehead = landmarks[FOREHEAD]

        nose_px = (int(nose.x * width), int(nose.y * height))
        chin_px = (int(chin.x * width), int(chin.y * height))
        left_eye_px = (int(left_eye.x * width), int(left_eye.y * height))
        right_eye_px = (int(right_eye.x * width), int(right_eye.y * height))
        forehead_px = (int(forehead.x * width), int(forehead.y * height))

        # Calculate gesture metrics
        eye_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
        horizontal_diff = nose_px[0] - eye_center_x
        nose_to_chin = chin_px[1] - nose_px[1]
        nose_to_forehead = nose_px[1] - forehead_px[1]

        # Calibration
        if not calibrated:
            calibration_data.append([horizontal_diff, nose_to_forehead, nose_to_chin])
            elapsed = time() - calibration_start

            cv2.putText(frame,
                        f"Calibrating... Hold neutral pose ({int(CALIBRATION_DURATION - elapsed)}s)",
                        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if elapsed >= CALIBRATION_DURATION:
                calibration_array = np.array(calibration_data)
                neutral_values["horizontal_diff"] = np.mean(calibration_array[:, 0])
                neutral_values["nose_to_forehead"] = np.mean(calibration_array[:, 1])
                neutral_values["nose_to_chin"] = np.mean(calibration_array[:, 2])
                calibrated = True
                print("✅ Calibration complete:", neutral_values)

        else:
            # Visual feedback bars
            draw_bar(frame, "↔ Left/Right", horizontal_diff,
                     neutral_values["horizontal_diff"], TOLERANCE["left_right"], 10, 50, (0, 0, 255))
            draw_bar(frame, "👆 Up", nose_to_forehead,
                     neutral_values["nose_to_forehead"], TOLERANCE["up"], 10, 90, (255, 0, 0))
            draw_bar(frame, "👇 Down", nose_to_chin,
                     neutral_values["nose_to_chin"], TOLERANCE["down"], 10, 130, (0, 255, 0))

            # Direction detection
            if horizontal_diff > neutral_values["horizontal_diff"] + TOLERANCE["left_right"]:
                trigger_action("right")
                direction_label = "Right"
            elif horizontal_diff < neutral_values["horizontal_diff"] - TOLERANCE["left_right"]:
                trigger_action("left")
                direction_label = "Left"
            elif nose_to_forehead < neutral_values["nose_to_forehead"] - TOLERANCE["up"]:
                trigger_action("up")
                direction_label = "Jump"
            elif nose_to_chin < neutral_values["nose_to_chin"] - TOLERANCE["down"]:
                trigger_action("down")
                direction_label = "Crouch"
            else:
                current_direction = None

            # Eye closure detection (skate)
            eye_top_y = landmarks[EYE_TOP].y * height
            eye_bottom_y = landmarks[EYE_BOTTOM].y * height
            eye_closure = eye_bottom_y - eye_top_y

            cv2.line(frame,
                     (int(landmarks[EYE_TOP].x * width), int(eye_top_y)),
                     (int(landmarks[EYE_BOTTOM].x * width), int(eye_bottom_y)),
                     (0, 255, 255), 2)
            cv2.putText(frame, f"EyeClosedDist: {eye_closure:.1f}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if eye_closure < EYE_CLOSED_THRESHOLD and not eye_closed_triggered:
                pyautogui.click()
                pyautogui.click()
                print("🛹 Skate Activated (Eye Closed)!")
                eye_closed_triggered = True
            elif eye_closure >= EYE_CLOSED_THRESHOLD:
                eye_closed_triggered = False

    else:
        cv2.putText(frame, "Face not detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # HUD
    cv2.putText(frame, f"Gesture: {direction_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Subway Surfers - Head & Eye Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
