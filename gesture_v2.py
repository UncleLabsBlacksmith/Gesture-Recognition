
import cv2
import mediapipe as mp
import pyautogui
import math

import time
# =====================================================
# AI VIRTUAL MOUSE 2.1
# Python + OpenCV + MediaPipe + PyAutoGUI
# =====================================================

# =====================================================
# 1. GENERAL SETTINGS
# =====================================================

SCREEN_W, SCREEN_H = pyautogui.size()

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.005

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

FRAME_MARGIN = 0.10

# =====================================================
# 2. CURSOR SETTINGS
# =====================================================

# Ignore minor cursor movements (screen pixels)
DEAD_ZONE_PX = 15

# One Euro Filter
MIN_CUTOFF = 1.2
BETA = 0.025
D_CUTOFF = 1.0

# Cursor speed
CURSOR_SPEED = 1.0

# Pinch / Drag settings
PINCH_START = 0.32
PINCH_RELEASE = 0.48

# Minimum movement required to start dragging
PINCH_DRAG_DEAD_ZONE = 28

# True = freeze cursor completely while pinching
# False = allow drag after passing drag dead zone
PINCH_LOCK_CURSOR = False

# Wait after pinch release
CURSOR_RESUME_DELAY = 0.15

# =====================================================
# 3. GESTURE SETTINGS
# =====================================================

GESTURE_COOLDOWN = 0.8

SCROLL_THRESHOLD = 0.012
SCROLL_SENSITIVITY = 100

HAND_LOST_TIMEOUT = 0.3

# =====================================================
# 4. MEDIAPIPE INITIALIZATION
# =====================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# =====================================================
# 5. WEBCAM
# =====================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open Webcam")

# =====================================================
# 6. ONE EURO FILTER
# =====================================================

class OneEuroFilter:

    def __init__(
        self,
        min_cutoff=1.2,
        beta=0.025,
        d_cutoff=1.0
    ):

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

        cutoff = (
            self.min_cutoff +
            self.beta * abs(dx_hat)
        )

        a = self.alpha(cutoff, dt)

        x_hat = (
            a * x +
            (1 - a) * self.prev_x
        )

        self.prev_x = x_hat
        self.prev_dx = dx_hat
        self.prev_time = now

        return x_hat

    def reset(self):
        self.prev_x = None
        self.prev_dx = 0
        self.prev_time = None


filter_x = OneEuroFilter(
    MIN_CUTOFF, BETA, D_CUTOFF
)

filter_y = OneEuroFilter(
    MIN_CUTOFF, BETA, D_CUTOFF
)

# =====================================================
# 7. SYSTEM STATES
# =====================================================

paused = False

pinching = False
mouse_down = False
drag_started = False

last_click_time = 0
last_gesture_time = 0
last_scroll_y = None

last_seen = time.time()

resume_cursor_time = 0

pinch_start_x = 0
pinch_start_y = 0

pinch_start_cursor_x = 0
pinch_start_cursor_y = 0

last_target_x = None
last_target_y = None

current_gesture = "WAITING"
current_action = "READY"

# =====================================================
# 8. HELPER FUNCTIONS
# =====================================================

def distance(a, b):

    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def finger_states(lm):

    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y

    return index, middle, ring, pinky


def map_to_screen(point, width, height):

    x = (point.x - FRAME_MARGIN) / (
        1 - 2 * FRAME_MARGIN
    )

    y = (point.y - FRAME_MARGIN) / (
        1 - 2 * FRAME_MARGIN
    )

    x = max(0, min(1, x))
    y = max(0, min(1, y))

    return (
        x * SCREEN_W,
        y * SCREEN_H
    )


def move_cursor_with_deadzone(x, y):

    global last_target_x, last_target_y

    global current_action

    if last_target_x is None:

        last_target_x = x
        last_target_y = y

        pyautogui.moveTo(
            int(x),
            int(y)
        )

        return

    dx = x - last_target_x
    dy = y - last_target_y

    movement = math.hypot(dx, dy)

    # Ignore minor movement
    if movement < DEAD_ZONE_PX:

        current_action = "MINOR MOVE - IGNORED"

        return

    # Move cursor
    current_x, current_y = pyautogui.position()

    new_x = current_x + dx * CURSOR_SPEED
    new_y = current_y + dy * CURSOR_SPEED

    new_x = max(0, min(SCREEN_W - 1, new_x))
    new_y = max(0, min(SCREEN_H - 1, new_y))

    pyautogui.moveTo(
        int(new_x),
        int(new_y)
    )

    last_target_x = x
    last_target_y = y

    current_action = "MOVE CURSOR"


def release_mouse():

    global mouse_down
    global pinching
    global drag_started

    if mouse_down:
        pyautogui.mouseUp()

    mouse_down = False
    pinching = False
    drag_started = False


def draw_interface(frame):

    h, w, _ = frame.shape

    color = (0, 0, 255) if paused else (0, 255, 0)

    cv2.rectangle(
        frame,
        (10, 10),
        (430, 165),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        "AI VIRTUAL MOUSE 2.1",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "GESTURE: " + current_gesture,
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    cv2.putText(
        frame,
        "ACTION: " + current_action,
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "DEAD ZONE: " + str(DEAD_ZONE_PX) + " px",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 200, 0),
        1
    )

    if paused:

        cv2.putText(
            frame,
            "PAUSED",
            (w - 220, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

# =====================================================
# 9. MAIN LOOP
# =====================================================

print("=" * 45)
print("AI VIRTUAL MOUSE 2.1")
print("=" * 45)

print("Q     = Quit")
print("SPACE = Pause / Resume")
print("ESC   = Quit")
print("=" * 45)

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
        current_action = "WAITING"

        if results.multi_hand_landmarks:

            last_seen = now

            for hand_landmarks in results.multi_hand_landmarks:

                lm = hand_landmarks.landmark

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # =====================================
                # LANDMARKS
                # =====================================

                index = lm[8]
                thumb = lm[4]

                i, m, r, p = finger_states(lm)

                palm_width = distance(lm[5], lm[17])

                pinch_ratio = (
                    distance(index, thumb) /
                    max(palm_width, 0.001)
                )

                # Convert index position
                target_x, target_y = map_to_screen(
                    index, w, h
                )

                # Apply One Euro Filter
                fx = filter_x.filter(target_x, now)
                fy = filter_y.filter(target_y, now)

                # =====================================
                # PINCH DETECTION
                # Highest priority
                # =====================================

                if pinching:

                    is_pinch = pinch_ratio < PINCH_RELEASE

                else:

                    is_pinch = pinch_ratio < PINCH_START

                if is_pinch:

                    current_gesture = "PINCH"

                    # Start pinch
                    if not pinching:

                        pinching = True
                        drag_started = False

                        pinch_start_x = fx
                        pinch_start_y = fy

                        pinch_start_cursor_x, pinch_start_cursor_y = (
                            pyautogui.position()
                        )

                        pyautogui.mouseDown()

                        mouse_down = True

                    # Movement during pinch
                    if PINCH_LOCK_CURSOR:

                        current_action = "CURSOR LOCKED"

                    else:

                        drag_distance = math.hypot(
                            fx - pinch_start_x,
                            fy - pinch_start_y
                        )

                        if drag_distance > PINCH_DRAG_DEAD_ZONE:

                            drag_started = True

                        if drag_started:

                            dx = fx - pinch_start_x
                            dy = fy - pinch_start_y

                            new_x = (
                                pinch_start_cursor_x +
                                dx * CURSOR_SPEED
                            )

                            new_y = (
                                pinch_start_cursor_y +
                                dy * CURSOR_SPEED
                            )

                            new_x = max(
                                0,
                                min(SCREEN_W - 1, new_x)
                            )

                            new_y = max(
                                0,
                                min(SCREEN_H - 1, new_y)
                            )

                            pyautogui.moveTo(
                                int(new_x),
                                int(new_y)
                            )

                            current_action = "DRAGGING"

                        else:

                            current_action = "PINCH - MICRO MOVE IGNORED"

                else:

                    # Release pinch
                    if pinching:

                        release_mouse()

                        resume_cursor_time = (
                            now + CURSOR_RESUME_DELAY
                        )

                        last_target_x = fx
                        last_target_y = fy

                        current_action = "CLICK / RELEASE"

                    # =================================
                    # PAUSE: FIST
                    # =================================

                    elif not any([i, m, r, p]):

                        paused = True

                        current_gesture = "FIST"
                        current_action = "PAUSED"

                        release_mouse()

                        last_scroll_y = None

                    # =================================
                    # RESUME: OPEN HAND
                    # =================================

                    elif all([i, m, r, p]):

                        paused = False

                        current_gesture = "OPEN HAND"
                        current_action = "RESUME"

                        last_scroll_y = None

                    # =================================
                    # RIGHT CLICK: THREE FINGERS
                    # =================================

                    elif i and m and r and not p:

                        current_gesture = "THREE FINGERS"
                        current_action = "RIGHT CLICK"

                        if not paused:

                            if now - last_gesture_time > GESTURE_COOLDOWN:

                                pyautogui.rightClick()

                                last_gesture_time = now

                        last_scroll_y = None

                    # =================================
                    # SCROLL: TWO FINGERS
                    # =================================

                    elif i and m and not r and not p:

                        current_gesture = "SCROLL MODE"

                        if not paused:

                            center_y = (
                                index.y + lm[12].y
                            ) / 2

                            if last_scroll_y is not None:

                                delta = last_scroll_y - center_y

                                if abs(delta) > SCROLL_THRESHOLD:

                                    amount = int(
                                        delta * SCROLL_SENSITIVITY
                                    )

                                    if amount != 0:

                                        pyautogui.scroll(amount)

                                        current_action = "SCROLL"

                            last_scroll_y = center_y

                        else:

                            current_action = "PAUSED"

                        last_scroll_y = center_y

                    # =================================
                    # THUMB UP: ENTER
                    # =================================

                    elif not i and not m and not r and p:

                        current_gesture = "THUMB UP"
                        current_action = "ENTER"

                        if not paused:

                            if now - last_gesture_time > GESTURE_COOLDOWN:

                                pyautogui.press("enter")

                                last_gesture_time = now

                        last_scroll_y = None

                    # =================================
                    # NORMAL MOUSE CONTROL
                    # =================================

                    elif i and not m and not r and not p:

                        current_gesture = "MOUSE CONTROL"

                        last_scroll_y = None

                        if not paused and now >= resume_cursor_time:

                            move_cursor_with_deadzone(
                                fx,
                                fy
                            )

                    else:

                        current_gesture = "OTHER GESTURE"

                        last_scroll_y = None

                        current_action = "WAITING"

        else:

            # =====================================
            # HAND LOST
            # =====================================

            if now - last_seen > HAND_LOST_TIMEOUT:

                if mouse_down:
                    release_mouse()

                pinching = False
                drag_started = False

                last_scroll_y = None

                current_gesture = "NO HAND"
                current_action = "WAITING"

        # =====================================
        # DISPLAY
        # =====================================

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

        draw_interface(frame)

        cv2.imshow(
            "AI Virtual Mouse 2.1",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:

            break

        elif key == ord(' '):

            paused = not paused

            if paused:
                release_mouse()

            current_action = (
                "PAUSED" if paused else "RESUMED"
            )

finally:

    # =============================================
    # SAFETY CLEANUP
    # =============================================

    try:
        release_mouse()
    except Exception:
        pass

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print("AI Virtual Mouse Stopped")
