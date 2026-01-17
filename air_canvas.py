import cv2
import mediapipe as mp
import numpy as np
import time

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Air Canvas", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# ================= VARIABLES =================
canvas = None
prev_x, prev_y = 0, 0
draw_color = (0, 0, 255)
brush_thickness = 6
eraser_thickness = 40

board_mode = "transparent"
last_switch = 0
SWITCH_DELAY = 1

# ================= FINGER LOGIC =================
def fingers_up(hand):
    tips = [8, 12, 16, 20]      # ignore thumb
    joints = [6, 10, 14, 18]
    fingers = []
    for tip, joint in zip(tips, joints):
        fingers.append(hand.landmark[tip].y < hand.landmark[joint].y)
    return fingers, fingers.count(True)

print("AIR CANVAS READY ✨")

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_landmarks = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        fingers, count = fingers_up(hand_landmarks)
        ix = int(hand_landmarks.landmark[8].x * w)
        iy = int(hand_landmarks.landmark[8].y * h)
        now = time.time()

        # ===== BOARD SWITCH =====
        if count == 4 and board_mode != "white" and now - last_switch > SWITCH_DELAY:
            board_mode = "white"
            last_switch = now

        elif count == 0 and board_mode != "transparent" and now - last_switch > SWITCH_DELAY:
            board_mode = "transparent"
            last_switch = now

        # ===== CLEAR =====
        if count == 5:
            canvas = np.zeros_like(frame)
            prev_x, prev_y = 0, 0

        # ===== COLOR SELECT =====
        elif fingers[0] and fingers[1] and iy < 80:
            if ix < 100:
                draw_color = (255, 0, 0)
            elif ix < 200:
                draw_color = (0, 255, 0)
            elif ix < 300:
                draw_color = (0, 0, 255)
            elif ix < 400:
                draw_color = (255, 0, 255)
            elif ix < 500:
                draw_color = (0, 0, 0)
            prev_x, prev_y = 0, 0

        # ===== DRAW =====
        elif fingers[0] and not fingers[1]:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = ix, iy
            thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness
            cv2.line(canvas, (prev_x, prev_y), (ix, iy), draw_color, thickness)
            prev_x, prev_y = ix, iy

        else:
            prev_x, prev_y = 0, 0

    # ================= BACKGROUND =================
    if board_mode == "white":
        base = np.ones_like(frame) * 255   # fully opaque white
    else:
        base = frame.copy()

    # ================= FINAL COMPOSE =================
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(base, base, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    final = cv2.add(bg, fg)


    # 🔥 DRAW HAND ON FINAL (THIS IS THE FIX)
    if hand_landmarks:
        mp_draw.draw_landmarks(
            final,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # ================= TOOLBAR =================
    cv2.rectangle(final, (0, 0), (w, 80), (40, 40, 40), -1)
    tools = [
        ((255, 0, 0), 0, 100),
        ((0, 255, 0), 100, 200),
        ((0, 0, 255), 200, 300),
        ((255, 0, 255), 300, 400),
        ((0, 0, 0), 400, 500)
    ]
    for color, x1, x2 in tools:
        cv2.rectangle(final, (x1, 0), (x2, 80), color, -1)

    cv2.putText(final, f"Board: {board_mode.upper()}",
                (550, 55), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.imshow("Air Canvas", final)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
