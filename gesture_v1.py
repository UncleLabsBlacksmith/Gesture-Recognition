
import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ==========================================
# AI VIRTUAL MOUSE 2.0
# ==========================================

# Screen
screen_w, screen_h = pyautogui.size()

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Cannot open Webcam")

# ==========================================
# ONE EURO FILTER
# ==========================================

class OneEuroFilter:

    def __init__(self, min_cutoff=1.2,
                 beta=0.025, d_cutoff=1.0):

        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.prev_x = None
        self.prev_dx = 0
        self.prev_time = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, now):

        if self.prev_x is None:
            self.prev_x = x
            self.prev_time = now
            return x

        dt = max(now - self.prev_time, 0.001)

        dx = (x - self.prev_x) / dt

        a_d = self.alpha(self.d_cutoff, dt)

        dx_hat = (
            a_d * dx +
            (1 - a_d) * self.prev_dx
        )

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        a = self.alpha(cutoff, dt)

        x_hat = (
            a * x +
            (1 - a) * self.prev_x
        )

        self.prev_x = x_hat
        self.prev_dx = dx_hat
        self.prev_time = now

        return x_hat


filter_x = OneEuroFilter()
filter_y = OneEuroFilter()

# ==========================================
# SETTINGS
# ==========================================

# Active camera area
FRAME_MARGIN = 0.10

# Pinch hysteresis
PINCH_START = 0.32
PINCH_RELEASE = 0.48

# Scroll sensitivity
SCROLL_SENSITIVITY = 80

# Gesture cooldown
GESTURE_COOLDOWN = 0.8

# ==========================================
# STATES
# ==========================================

mouse_down = False
pinching = False
paused = False

last_scroll_y = None
last_scroll_time = 0
last_gesture_time = 0
last_seen = time.time()

smooth_x = screen_w / 2
smooth_y = screen_h / 2

current_gesture = "WAITING"
current_action = "READY"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(a, b):

    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def finger_states(lm):

    # Finger tips and PIP joints
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    return index, middle, ring, pinky


def draw_status(frame, gesture, action, paused):

    color = (0, 0, 255) if paused else (0, 255, 0)

    cv2.rectangle(
        frame,
        (10, 10),
        (410, 135),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "AI VIRTUAL MOUSE 2.0",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "GESTURE: " + gesture,
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    cv2.putText(
        frame,
        "ACTION: " + action,
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    if paused:
        cv2.putText(
            frame,
            "PAUSED",
            (450, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )


# ==========================================
# MAIN LOOP
# ==========================================

print("AI Virtual Mouse 2.0 Started")
print("Press Q to quit")
print("Press SPACE to pause/resume")

try:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        now = time.time()

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        current_gesture = "NO HAND"
        current_action = "READY"

        if results.multi_hand_landmarks:

            last_seen = now

            for hand_landmarks in results.multi_hand_landmarks:

                lm = hand_landmarks.landmark

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                index = lm[8]
                thumb = lm[4]

                # Finger states
                i, m, r, p = finger_states(lm)

                # Palm scale
                palm_width = distance(lm[5], lm[17])

                # Pinch ratio
                pinch_ratio = (
                    distance(index, thumb) /
                    max(palm_width, 0.001)
                )

                # ==================================
                # PAUSE: FIST
                # ==================================

                if not any([i, m, r, p]):

                    paused = True
                    current_gesture = "FIST"
                    current_action = "PAUSED"

                    if mouse_down:
                        pyautogui.mouseUp()
                        mouse_down = False

                    pinching = False
                    last_scroll_y = None

                # ==================================
                # RESUME: OPEN HAND
                # ==================================

                elif all([i, m, r, p]):

                    paused = False
                    current_gesture = "OPEN HAND"
                    current_action = "RESUME"

                    last_scroll_y = None

                # ==================================
                # ACTIVE CONTROL
                # ==================================

                else:

                    # ==================================
                    # RIGHT CLICK: THREE FINGERS
                    # ==================================

                    if i and m and r and not p:

                        current_gesture = "THREE FINGERS"
                        current_action = "RIGHT CLICK"

                        if not paused and now - last_gesture_time > GESTURE_COOLDOWN:

                            if mouse_down:
                                pyautogui.mouseUp()
                                mouse_down = False

                            pyautogui.rightClick()
                            last_gesture_time = now

                        last_scroll_y = None

                    # ==================================
                    # SCROLL: TWO FINGERS
                    # ==================================

                    elif i and m and not r and not p:

                        current_gesture = "SCROLL MODE"

                        if not paused:

                            center_y = (index.y + lm[12].y) / 2

                            if last_scroll_y is not None:

                                delta = last_scroll_y - center_y

                                if abs(delta) > 0.012:

                                    amount = int(
                                        delta * SCROLL_SENSITIVITY
                                    )

                                    if amount != 0:
                                        pyautogui.scroll(amount)

                                        current_action = "SCROLL"

                            last_scroll_y = center_y

                        if mouse_down:
                            pyautogui.mouseUp()
                            mouse_down = False
                            pinching = False

                    # ==================================
                    # THUMB UP: ENTER
                    # ==================================

                    elif not i and not m and not r and p:

                        current_gesture = "THUMB UP"
                        current_action = "ENTER"

                        if not paused and now - last_gesture_time > GESTURE_COOLDOWN:

                            if mouse_down:
                                pyautogui.mouseUp()
                                mouse_down = False

                            pyautogui.press("enter")
                            last_gesture_time = now

                        last_scroll_y = None

                    # ==================================
                    # NORMAL MOUSE CONTROL
                    # ==================================

                    else:

                        last_scroll_y = None

                        if not paused:

                            current_gesture = "MOUSE CONTROL"

                            # Map index finger to screen
                            x = (
                                index.x - FRAME_MARGIN
                            ) / (1 - 2 * FRAME_MARGIN)

                            y = (
                                index.y - FRAME_MARGIN
                            ) / (1 - 2 * FRAME_MARGIN)

                            target_x = max(
                                0,
                                min(screen_w - 1, x * screen_w)
                            )

                            target_y = max(
                                0,
                                min(screen_h - 1, y * screen_h)
                            )

                            # One Euro filtering
                            fx = filter_x.filter(target_x, now)
                            fy = filter_y.filter(target_y, now)

                            pyautogui.moveTo(
                                int(fx),
                                int(fy)
                            )

                            # ==================================
                            # LEFT CLICK / DRAG
                            # ==================================

                            if pinch_ratio < PINCH_START:

                                current_gesture = "PINCH"
                                current_action = "DRAG / LEFT CLICK"

                                if not pinching:

                                    pinching = True

                                    if not mouse_down:
                                        pyautogui.mouseDown()
                                        mouse_down = True

                            elif pinch_ratio > PINCH_RELEASE:

                                if pinching:

                                    if mouse_down:
                                        pyautogui.mouseUp()
                                        mouse_down = False

                                    pinching = False

                                    current_action = "RELEASE"

                                else:
                                    current_action = "MOVE CURSOR"

        else:

            # Release mouse if hand disappears
            if now - last_seen > 0.3:

                if mouse_down:
                    pyautogui.mouseUp()
                    mouse_down = False

                pinching = False
                last_scroll_y = None

        # ==================================
        # DISPLAY
        # ==================================

        draw_status(
            frame,
            current_gesture,
            current_action,
            paused
        )

        # Active area
        x1 = int(w * FRAME_MARGIN)
        y1 = int(h * FRAME_MARGIN)
        x2 = int(w * (1 - FRAME_MARGIN))
        y2 = int(h * (1 - FRAME_MARGIN))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 150, 0),
            1
        )

        cv2.imshow(
            "AI Virtual Mouse 2.0",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            break

        elif key == ord(' '):

            paused = not paused

            if paused and mouse_down:
                pyautogui.mouseUp()
                mouse_down = False
                pinching = False

finally:

    # Safety: always release mouse
    if mouse_down:
        pyautogui.mouseUp()

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("AI Virtual Mouse Stopped")
