
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Screen settings
screen_w, screen_h = pyautogui.size()

# Safety
pyautogui.FAILSAFE = True

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

# Mouse smoothing
smooth_x, smooth_y = 0, 0
smooth_factor = 0.3

# Click control
pinch_active = False
last_click = 0
click_delay = 0.5

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            index = hand_landmarks.landmark[8]

            # Thumb tip
            thumb = hand_landmarks.landmark[4]

            # Map camera coordinates to screen
            target_x = int(index.x * screen_w)
            target_y = int(index.y * screen_h)

            # Smooth cursor movement
            smooth_x += (target_x - smooth_x) * smooth_factor
            smooth_y += (target_y - smooth_y) * smooth_factor

            pyautogui.moveTo(
                int(smooth_x),
                int(smooth_y)
            )

            # Distance between thumb and index
            distance = math.hypot(
                index.x - thumb.x,
                index.y - thumb.y
            )

            # Pinch detection
            if distance < 0.045:

                cv2.putText(
                    frame,
                    "PINCH / CLICK",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                if not pinch_active:
                    if time.time() - last_click > click_delay:
                        pyautogui.click()
                        last_click = time.time()

                pinch_active = True

            else:
                pinch_active = False

    cv2.imshow("AI Hand Tracking Mouse", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
