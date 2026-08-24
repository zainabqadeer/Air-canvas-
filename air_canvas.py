import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os

# -----------------------------
# DOWNLOAD MODEL IF NEEDED
# -----------------------------

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~8MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded! Starting...")

# -----------------------------
# MEDIAPIPE HAND CONNECTIONS
# (21 landmarks, standard connections)
# -----------------------------

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Thumb
    (0,5),(5,6),(6,7),(7,8),       # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16),# Ring
    (0,17),(17,18),(18,19),(19,20),# Pinky
    (5,9),(9,13),(13,17)           # Palm
]

# -----------------------------
# SETUP
# -----------------------------

BaseOptions      = mp_python.BaseOptions
HandLandmarker   = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
canvas   = None
prev_x, prev_y = 0, 0
draw_color = (255, 0, 0)   # BGR: Blue

buttons = [
    ("BLUE",   (50,  20, 170, 70), (255, 0,   0)),
    ("GREEN",  (190, 20, 310, 70), (0,   200, 0)),
    ("RED",    (330, 20, 450, 70), (0,   0,   255)),
    ("ERASER", (470, 20, 620, 70), (0,   0,   0)),
]

# -----------------------------
# HELPERS
# -----------------------------

def fingers_up(lm, h, w):
    """Returns [index_up, middle_up] as booleans."""
    tips = [8, 12]
    pips = [6, 10]
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]


def draw_skeleton(frame, landmarks, h, w):
    """Draw hand landmarks and connections manually."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
    for i, pt in enumerate(pts):
        r = 6 if i in [4, 8, 12, 16, 20] else 3
        cv2.circle(frame, pt, r, (255, 255, 255), cv2.FILLED)
        cv2.circle(frame, pt, r, (0, 180, 100), 1)


def draw_buttons(frame):
    for name, (x1, y1, x2, y2), color in buttons:
        is_eraser = name == "ERASER"
        is_active = (draw_color == (0,0,0) and is_eraser) or (not is_eraser and draw_color == color)

        btn_color = (240, 240, 240) if is_eraser else color
        txt_color = (0, 0, 0)       if is_eraser else (255, 255, 255)

        # Shadow
        cv2.rectangle(frame, (x1+3, y1+3), (x2+3, y2+3), (20,20,20), cv2.FILLED)
        # Button
        cv2.rectangle(frame, (x1, y1), (x2, y2), btn_color, cv2.FILLED)
        # Active border
        if is_active:
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 3)

        cv2.putText(frame, name, (x1+10, y1+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)


# -----------------------------
# MAIN LOOP
# -----------------------------

frame_ts_ms = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Run detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts_ms += 33
        result = landmarker.detect_for_video(mp_img, frame_ts_ms)

        # Draw UI buttons
        draw_buttons(frame)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            draw_skeleton(frame, lm, h, w)

            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            index_up, middle_up = fingers_up(lm, h, w)

            # ---- SELECTION MODE (✌ 2 fingers) ----
            if index_up and middle_up:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (ix, iy), 14, (255, 255, 255), 2)

                if 20 < iy < 70:
                    for name, (x1, y1, x2, y2), color in buttons:
                        if x1 < ix < x2:
                            draw_color = (0, 0, 0) if name == "ERASER" else color

            # ---- DRAWING MODE (☝ 1 finger) ----
            elif index_up and not middle_up:
                if iy > 80:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = ix, iy
                    thickness = 30 if draw_color == (0, 0, 0) else 8
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), draw_color, thickness)
                    prev_x, prev_y = ix, iy

                    dot_c = (180, 180, 180) if draw_color == (0,0,0) else draw_color
                    cv2.circle(frame, (ix, iy), thickness // 2, dot_c, cv2.FILLED)
            else:
                prev_x, prev_y = 0, 0

        # ---- MERGE CANVAS + CAMERA ----
        gray   = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask     = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv    = cv2.bitwise_not(mask)
        frame_bg    = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
        drawing_fg  = cv2.bitwise_and(canvas, canvas, mask=mask)
        output      = cv2.add(frame_bg, drawing_fg)

        # ---- INSTRUCTION BAR ----
        cv2.rectangle(output, (0, h-38), (w, h), (25, 25, 25), cv2.FILLED)
        cv2.putText(output,
            "  [1 finger] Draw    [2 fingers] Pick color    [C] Clear    [Q] Quit",
            (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)

        cv2.imshow("AirCanvas - Virtual Drawing Board", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Bye!")
