"""
AI VIRTUAL MOUSE 3.0
Python + OpenCV + MediaPipe  (Windows)

Gestures
--------
POINT   (index up)                 -> move cursor
PINCH   (thumb + index)            -> press (click on release) | move = drag
RCLICK  (thumb + middle pinch)     -> right click
SCROLL  (index + middle up)        -> move hand up/down to scroll
THUMB   (thumb up, fist closed)    -> Enter
FIST    (hold ~0.8 s)              -> pause
OPEN    (hold ~0.6 s while paused) -> resume

Keys: Q / ESC quit | SPACE pause | L toggle landmarks | +/- cursor smoothing
"""

import ctypes
import math
import sys
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import pyautogui

# =====================================================
# 1. SETTINGS
# =====================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Active area of the camera frame that maps to the whole screen
MARGIN_X = 0.12
MARGIN_Y = 0.12

# --- Cursor smoothing (One Euro Filter) ---
MIN_CUTOFF = 1.0     # lower = smoother when slow (more lag)
BETA = 0.03          # higher = less lag when moving fast
D_CUTOFF = 1.0
JITTER_PX = 2        # ignore sub-pixel shake only (no "stepping")

# --- Pinch (ratio = thumb-finger distance / palm width) ---
PINCH_START = 0.32
PINCH_RELEASE = 0.48
MID_PINCH_START = 0.40       # right-click pinch (looser than left)
PINCH_FREEZE = 0.45          # freeze cursor when fingers approach each other
INDEX_REACH_MIN = 1.12       # index must be extended to start a pinch (rejects fist)
DRAG_THRESHOLD_PX = 28       # movement needed before the cursor follows the hand
CURSOR_RESUME_DELAY = 0.15   # seconds the cursor stays frozen after a pinch ends

# --- Gestures ---
STABLE_FRAMES = 2            # frames a gesture must hold before it counts
RIGHT_CLICK_COOLDOWN = 0.5
ENTER_COOLDOWN = 1.0
PAUSE_HOLD = 0.8             # fist hold time -> pause
RESUME_HOLD = 0.6            # open palm hold time -> resume
ENABLE_THUMB_ENTER = True

# --- Scroll ---
SCROLL_DEADBAND = 0.003      # normalised hand movement ignored per frame
SCROLL_GAIN = 9000           # wheel units per 1.0 of normalised movement

HAND_LOST_TIMEOUT = 0.3
SHOW_LANDMARKS = True

# =====================================================
# 2. WINDOWS MOUSE (fast, falls back to pyautogui)
# =====================================================

pyautogui.FAILSAFE = False   # corners are reachable by design; use Q/ESC to quit
pyautogui.PAUSE = 0

try:
    ctypes.windll.user32.SetProcessDPIAware()
    _user32 = ctypes.windll.user32
except Exception:
    _user32 = None

SCREEN_W, SCREEN_H = pyautogui.size()


class Mouse:
    LEFTDOWN, LEFTUP = 0x0002, 0x0004
    RIGHTDOWN, RIGHTUP = 0x0008, 0x0010
    WHEEL = 0x0800

    @staticmethod
    def position():
        return pyautogui.position()

    @staticmethod
    def move(x, y):
        x = int(max(0, min(SCREEN_W - 1, x)))
        y = int(max(0, min(SCREEN_H - 1, y)))
        if _user32:
            _user32.SetCursorPos(x, y)
        else:
            pyautogui.moveTo(x, y)

    @classmethod
    def _event(cls, flag, data=0):
        if _user32:
            _user32.mouse_event(flag, 0, 0, data, 0)
            return True
        return False

    @classmethod
    def left_down(cls):
        if not cls._event(cls.LEFTDOWN):
            pyautogui.mouseDown()

    @classmethod
    def left_up(cls):
        if not cls._event(cls.LEFTUP):
            pyautogui.mouseUp()

    @classmethod
    def right_click(cls):
        if not (cls._event(cls.RIGHTDOWN) and cls._event(cls.RIGHTUP)):
            pyautogui.rightClick()

    @classmethod
    def scroll(cls, amount):
        amount = int(amount)
        if amount == 0:
            return
        if not cls._event(cls.WHEEL, amount & 0xFFFFFFFF if amount < 0 else amount):
            pyautogui.scroll(amount)


# =====================================================
# 3. THREADED CAMERA (no blocking in the main loop)
# =====================================================

class Camera:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        self.lock = threading.Lock()
        self.frame = None
        self.counter = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.counter += 1

    def read(self):
        with self.lock:
            return self.counter, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


# =====================================================
# 4. ONE EURO FILTER
# =====================================================

class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.prev_x = None
        self.prev_dx = 0.0
        self.prev_t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.prev_x is None:
            self.prev_x, self.prev_t = x, t
            return x

        dt = max(t - self.prev_t, 1e-3)
        dx = (x - self.prev_x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.prev_dx

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.prev_x

        self.prev_x, self.prev_dx, self.prev_t = x_hat, dx_hat, t
        return x_hat


# =====================================================
# 5. LANDMARK HELPERS
# =====================================================

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_up(lm, tip, pip):
    """Rotation-invariant: tip is farther from the wrist than the PIP joint."""
    return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]) * 1.05


def thumb_extended(lm, palm):
    return dist(lm[4], lm[5]) / palm > 0.6


def thumb_pointing_up(lm, palm):
    """Thumb extended AND pointing upward (tolerates a tilted hand)."""
    return lm[4].y < lm[3].y < lm[2].y and lm[4].y < lm[5].y - 0.3 * palm


# =====================================================
# 6. CONTROLLER
# =====================================================

class VirtualMouse:
    def __init__(self):
        self.fx = OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)
        self.fy = OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)

        self.paused = False
        self.gesture = "NO HAND"
        self.action = "WAITING"
        self.pinch_ratio = 1.0

        self.history = deque(maxlen=STABLE_FRAMES)
        self.stable = "NONE"

        # cursor
        self.cursor = Mouse.position()
        self.cursor = (float(self.cursor[0]), float(self.cursor[1]))
        self.resume_time = 0.0

        # left pinch
        self.idx_pinch = False
        self.pinching = False
        self.mouse_down = False
        self.dragging = False
        self.anchor_f = (0.0, 0.0)
        self.anchor_cursor = (0.0, 0.0)

        # right pinch
        self.mid_pinch = False
        self.last_rclick = 0.0

        # misc
        self.last_enter = 0.0
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.fist_since = None
        self.open_since = None
        self.latch = False        # fist/open toggle must be released first
        self.last_seen = time.time()
        self.toast = ("", 0.0)

    # ---------- utilities ----------

    def say(self, text, now):
        self.toast = (text, now)

    def set_cursor(self, x, y):
        x = max(0, min(SCREEN_W - 1, x))
        y = max(0, min(SCREEN_H - 1, y))
        if math.hypot(x - self.cursor[0], y - self.cursor[1]) < JITTER_PX:
            return
        self.cursor = (x, y)
        Mouse.move(x, y)

    def release_all(self):
        if self.mouse_down:
            Mouse.left_up()
        self.mouse_down = False
        self.pinching = False
        self.dragging = False
        self.idx_pinch = False
        self.mid_pinch = False
        self.scroll_y = None
        self.scroll_acc = 0.0

    def set_paused(self, value, now):
        self.paused = value
        self.release_all()
        self.latch = True
        self.say("PAUSED" if value else "RESUMED", now)

    def lost(self, now):
        if now - self.last_seen > HAND_LOST_TIMEOUT:
            self.release_all()
            self.fx.reset()
            self.fy.reset()
            self.history.clear()
            self.stable = "NONE"
            self.fist_since = self.open_since = None
            self.gesture, self.action = "NO HAND", "WAITING"

    # ---------- main update ----------

    def update(self, lm, now):
        self.last_seen = now
        palm = max(dist(lm[5], lm[17]), 1e-3)

        r_idx = dist(lm[4], lm[8]) / palm
        r_mid = dist(lm[4], lm[12]) / palm
        self.pinch_ratio = r_idx

        idx = finger_up(lm, 8, 6)
        mid = finger_up(lm, 12, 10)
        ring = finger_up(lm, 16, 14)
        pinky = finger_up(lm, 20, 18)
        thumb = thumb_extended(lm, palm)
        four = (idx, mid, ring, pinky)

        # pointer = blend of index knuckle and tip (stable while pinching)
        px = lm[5].x * 0.35 + lm[8].x * 0.65
        py = lm[5].y * 0.35 + lm[8].y * 0.65
        sx = (px - MARGIN_X) / (1 - 2 * MARGIN_X)
        sy = (py - MARGIN_Y) / (1 - 2 * MARGIN_Y)
        sx = max(0.0, min(1.0, sx)) * (SCREEN_W - 1)
        sy = max(0.0, min(1.0, sy)) * (SCREEN_H - 1)
        fx, fy = self.fx(sx, now), self.fy(sy, now)

        # ----- classify static gesture -----
        if not any(four):
            raw = "THUMB" if (thumb and thumb_pointing_up(lm, palm)) else "FIST"
        elif all(four):
            raw = "OPEN"
        elif idx and mid and not ring and not pinky:
            raw = "SCROLL"
        elif idx and not (mid or ring or pinky):
            raw = "POINT"
        else:
            raw = "NONE"

        self.history.append(raw)
        if len(self.history) == STABLE_FRAMES and len(set(self.history)) == 1:
            self.stable = raw
        g = self.stable

        # ----- pause / resume (always evaluated) -----
        if g == "FIST":
            self.open_since = None
            if self.fist_since is None:
                self.fist_since = now
            if not self.paused and not self.latch and now - self.fist_since > PAUSE_HOLD:
                self.set_paused(True, now)
        elif g == "OPEN":
            self.fist_since = None
            if self.open_since is None:
                self.open_since = now
            if self.paused and not self.latch and now - self.open_since > RESUME_HOLD:
                self.set_paused(False, now)
        else:
            self.fist_since = self.open_since = None
            self.latch = False

        if self.paused:
            self.gesture = g if g != "NONE" else "..."
            self.action = "PAUSED (open palm to resume)"
            return

        # ----- pinch hysteresis -----
        index_reach = dist(lm[8], lm[0]) / max(dist(lm[5], lm[0]), 1e-3)
        if self.idx_pinch:
            self.idx_pinch = r_idx < PINCH_RELEASE
        else:
            self.idx_pinch = r_idx < PINCH_START

        if self.mid_pinch:
            self.mid_pinch = r_mid < PINCH_RELEASE and not self.idx_pinch
        else:
            self.mid_pinch = (
                r_mid < MID_PINCH_START
                and r_mid < r_idx
                and r_idx > PINCH_START * 1.4
            )

        near_pinch = min(r_idx, r_mid) < PINCH_FREEZE and (idx or mid)

        # ----- LEFT PINCH: click / drag -----
        if self.idx_pinch:
            self.gesture = "PINCH"
            if not self.pinching:
                self.pinching = True
                self.dragging = False
                self.anchor_f = (fx, fy)
                self.anchor_cursor = self.cursor
                Mouse.left_down()          # same as v2: press immediately
                self.mouse_down = True
                self.action = "PINCH"
            else:
                dx, dy = fx - self.anchor_f[0], fy - self.anchor_f[1]
                if not self.dragging and math.hypot(dx, dy) > DRAG_THRESHOLD_PX:
                    self.dragging = True
                if self.dragging:
                    self.cursor = (-1e9, -1e9)   # force move
                    self.set_cursor(
                        self.anchor_cursor[0] + dx, self.anchor_cursor[1] + dy
                    )
                    self.action = "DRAGGING"
                else:
                    self.action = "PINCH"
            return

        if self.pinching:
            Mouse.left_up()
            self.mouse_down = False
            self.action = "DROP" if self.dragging else "CLICK"
            self.say(self.action, now)
            self.pinching = False
            self.dragging = False
            self.resume_time = now + CURSOR_RESUME_DELAY
            self.gesture = "RELEASE"
            return

        # ----- RIGHT PINCH -----
        if self.mid_pinch:
            self.gesture = "RCLICK"
            if not getattr(self, "_mid_fired", False):
                self._mid_fired = True
                if now - self.last_rclick > RIGHT_CLICK_COOLDOWN:
                    Mouse.right_click()
                    self.last_rclick = now
                    self.say("RIGHT CLICK", now)
            self.action = "RIGHT CLICK"
            return
        self._mid_fired = False

        # ----- other gestures -----
        self.gesture = g
        if g != "SCROLL":
            self.scroll_y = None
            self.scroll_acc = 0.0

        if g == "POINT":
            if now >= self.resume_time and not near_pinch:
                self.set_cursor(fx, fy)
                self.action = "MOVE"
            else:
                self.action = "HOLD"

        elif g == "SCROLL":
            y = (lm[8].y + lm[12].y) / 2
            if self.scroll_y is not None:
                delta = self.scroll_y - y
                if abs(delta) > SCROLL_DEADBAND:
                    self.scroll_acc += delta * SCROLL_GAIN
                    step = int(self.scroll_acc)
                    if abs(step) >= 10:
                        Mouse.scroll(step)
                        self.scroll_acc -= step
                        self.action = "SCROLL UP" if step > 0 else "SCROLL DOWN"
                else:
                    self.action = "SCROLL"
            self.scroll_y = y

        elif g == "THUMB" and ENABLE_THUMB_ENTER:
            if now - self.last_enter > ENTER_COOLDOWN:
                pyautogui.press("enter")
                self.last_enter = now
                self.say("ENTER", now)
            self.action = "ENTER"

        elif g == "FIST":
            self.action = "HOLD TO PAUSE"
        elif g == "OPEN":
            self.action = "READY"
        else:
            self.action = "WAITING"


# =====================================================
# 7. UI
# =====================================================

def draw_ui(frame, vm, fps, now):
    h, w, _ = frame.shape

    x1, y1 = int(w * MARGIN_X), int(h * MARGIN_Y)
    x2, y2 = int(w * (1 - MARGIN_X)), int(h * (1 - MARGIN_Y))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 1)

    # translucent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    state_color = (0, 0, 255) if vm.paused else (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "AI VIRTUAL MOUSE 3.0", (22, 38), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"GESTURE: {vm.gesture}", (22, 68), font, 0.55, state_color, 2)
    cv2.putText(frame, f"ACTION : {vm.action}", (22, 94), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (22, 120), font, 0.5, (255, 200, 0), 1)

    # pinch meter
    bx, by, bw = 22, 134, 200
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 8), (70, 70, 70), -1)
    fill = int(bw * max(0.0, min(1.0, 1 - vm.pinch_ratio / 1.2)))
    cv2.rectangle(
        frame, (bx, by), (bx + fill, by + 8),
        (0, 200, 255) if vm.pinching else (0, 255, 0), -1
    )
    thr = int(bw * (1 - PINCH_START / 1.2))
    cv2.line(frame, (bx + thr, by - 3), (bx + thr, by + 11), (255, 255, 255), 1)

    if vm.paused:
        cv2.putText(frame, "PAUSED", (w - 230, 60), font, 1.1, (0, 0, 255), 3)

    text, t = vm.toast
    if text and now - t < 0.8:
        size = cv2.getTextSize(text, font, 1.2, 3)[0]
        cv2.putText(
            frame, text, ((w - size[0]) // 2, h - 40), font, 1.2, (0, 255, 255), 3
        )


# =====================================================
# 8. MAIN
# =====================================================

def main():
    global SHOW_LANDMARKS, MIN_CUTOFF

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cam = Camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    vm = VirtualMouse()

    print("=" * 45)
    print("AI VIRTUAL MOUSE 3.0")
    print("=" * 45)
    print("Q/ESC quit | SPACE pause | L landmarks | +/- smoothing")
    print("=" * 45)

    last_counter = 0
    last_new_frame = time.time()
    fps, prev = 0.0, time.time()

    try:
        while True:
            counter, frame = cam.read()
            now = time.time()

            if frame is None or counter == last_counter:
                if now - last_new_frame > 3:
                    print("Camera stopped delivering frames.")
                    break
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            last_counter = counter
            last_new_frame = now
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)
            prev = now

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                if SHOW_LANDMARKS:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                vm.update(hand.landmark, now)
            else:
                vm.lost(now)

            draw_ui(frame, vm, fps, now)
            cv2.imshow("AI Virtual Mouse 3.0", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                vm.set_paused(not vm.paused, now)
            elif key == ord("l"):
                SHOW_LANDMARKS = not SHOW_LANDMARKS
            elif key in (ord("+"), ord("=")):
                vm.fx.min_cutoff = vm.fy.min_cutoff = min(5.0, vm.fx.min_cutoff + 0.2)
                vm.say(f"CUTOFF {vm.fx.min_cutoff:.1f} (less smooth)", now)
            elif key in (ord("-"), ord("_")):
                vm.fx.min_cutoff = vm.fy.min_cutoff = max(0.2, vm.fx.min_cutoff - 0.2)
                vm.say(f"CUTOFF {vm.fx.min_cutoff:.1f} (smoother)", now)
    finally:
        try:
            vm.release_all()
        except Exception:
            pass
        cam.release()
        hands.close()
        cv2.destroyAllWindows()
        print("AI Virtual Mouse Stopped")


if __name__ == "__main__":
    sys.exit(main())
