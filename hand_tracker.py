"""
Hand Tracker - Camera முன்னாடி கை காட்டினால்:
  - Hand Color   (கை நிறம்)
  - Finger Count (விரல் எண்ணிக்கை)
  - Movement     (நகரும் திசை)
  - Position     (திரையில் இடம்)

Install:
    pip install opencv-python mediapipe numpy

Run:
    python hand_tracker.py
"""

import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

# ── Version detect & setup ────────────────────────────────────────────────────
try:
    _hands_test = mp.solutions.hands
    USE_LEGACY  = True
    print("[OK] MediaPipe legacy solutions API found")
except AttributeError:
    USE_LEGACY  = False
    print("[OK] MediaPipe new Task API mode")

FINGERTIP_IDS  = [4, 8, 12, 16, 20]
FINGER_PIP_IDS = [3, 6, 10, 14, 18]
wrist_history  = deque(maxlen=14)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# ── Helper functions ──────────────────────────────────────────────────────────

def detect_colour(hsv, lms, shape):
    h, w = shape[:2]
    pts  = [0,1,5,9,13,17]
    xs   = [int(lms[i].x * w) for i in pts]
    ys   = [int(lms[i].y * h) for i in pts]
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    r = 14
    patch = hsv[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
    if patch.size == 0:
        return "Unknown", (200,200,200)
    s_mean = float(np.mean(patch[:,:,1]))
    v_mean = float(np.mean(patch[:,:,2]))
    if s_mean < 40:
        if v_mean > 180: return "Pale/White",  (220,200,190)
        elif v_mean > 80: return "Brown/Gray", (130,100,80)
        else:             return "Dark",        (60, 40, 30)
    hue = int(np.median(patch[:,:,0].flatten()))
    TABLE = [
        ("Red",   0,10,(0,0,220)),("Orange",11,25,(0,140,255)),
        ("Yellow",26,35,(0,220,220)),("Green",36,85,(0,200,0)),
        ("Cyan",  86,100,(200,200,0)),("Blue",101,130,(220,100,0)),
        ("Purple",131,160,(180,0,180)),("Pink",161,175,(180,0,120)),
        ("Red",   176,180,(0,0,200)),
    ]
    for name,lo,hi,bgr in TABLE:
        if lo <= hue <= hi:
            return name, bgr
    return "Skin", (100,180,220)


def count_fingers(lms, label):
    count = 0
    # Thumb
    if label == "Right":
        if lms[4].x < lms[3].x: count += 1
    else:
        if lms[4].x > lms[3].x: count += 1
    # Other 4
    for tip,pip in zip(FINGERTIP_IDS[1:], FINGER_PIP_IDS[1:]):
        if lms[tip].y < lms[pip].y:
            count += 1
    return count


def get_movement(wx, wy):
    if len(wrist_history) < 6:
        return "Still", (200,200,200)
    dx = wx - wrist_history[0][0]
    dy = wy - wrist_history[0][1]
    if np.hypot(dx,dy) < 9:
        return "Still", (200,200,200)
    ang = np.degrees(np.arctan2(-dy, dx))
    if   -45 <= ang <  45:  return "Right ->", (0,200,255)
    elif  45 <= ang < 135:  return "Up ^",     (0,255,150)
    elif ang >= 135 or ang < -135: return "Left <-", (255,150,0)
    else:                   return "Down v",   (0,100,255)


def get_position(nx, ny):
    col = "Left" if nx<0.33 else ("Center" if nx<0.66 else "Right")
    row = "Top"  if ny<0.33 else ("Mid"    if ny<0.66 else "Bottom")
    return f"{row}-{col}"


def draw_panel(frame, rows, x=10, y=10):
    pw, ph = 310, len(rows)*34+14
    ov = frame.copy()
    cv2.rectangle(ov,(x,y),(x+pw,y+ph),(20,20,20),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    for i,(lbl,val,col) in enumerate(rows):
        ty = y+30+i*34
        cv2.putText(frame,f"{lbl}:",(x+10,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(170,170,170),1,cv2.LINE_AA)
        cv2.putText(frame,str(val),(x+140,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,col,2,cv2.LINE_AA)


def draw_finger_dots(frame, lms, w, h):
    raised = []
    if lms[4].x < lms[3].x or lms[4].x > lms[3].x:
        raised.append(4)
    for tip,pip in zip(FINGERTIP_IDS[1:],FINGER_PIP_IDS[1:]):
        if lms[tip].y < lms[pip].y:
            raised.append(tip)
    for tid in FINGERTIP_IDS:
        px,py = int(lms[tid].x*w), int(lms[tid].y*h)
        col = (0,255,120) if tid in raised else (60,60,200)
        cv2.circle(frame,(px,py),11,col,-1)
        cv2.circle(frame,(px,py),11,(255,255,255),1)


def draw_skeleton(frame, lms, w, h):
    for a,b in HAND_CONNECTIONS:
        p1 = (int(lms[a].x*w), int(lms[a].y*h))
        p2 = (int(lms[b].x*w), int(lms[b].y*h))
        cv2.line(frame, p1, p2, (80,200,120), 2)
    for lm in lms:
        cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),4,(255,255,255),-1)


def process_hand(frame, hsv, lms, label, idx):
    h, w = frame.shape[:2]
    wx,wy = int(lms[0].x*w), int(lms[0].y*h)
    wrist_history.append((wx,wy))

    col_name, col_bgr = detect_colour(hsv, lms, frame.shape)
    fingers           = count_fingers(lms, label)
    move, mc          = get_movement(wx, wy)
    pos               = get_position(lms[0].x, lms[0].y)

    draw_skeleton(frame, lms, w, h)
    draw_finger_dots(frame, lms, w, h)

    xs = [int(lm.x*w) for lm in lms]
    ys = [int(lm.y*h) for lm in lms]
    x1,y1 = max(0,min(xs)-20), max(0,min(ys)-20)
    x2,y2 = min(w,max(xs)+20), min(h,max(ys)+20)
    cv2.rectangle(frame,(x1,y1),(x2,y2),col_bgr,2)
    cv2.putText(frame,str(fingers),(max(5,x1-38),y1+52),
                cv2.FONT_HERSHEY_SIMPLEX,1.8,(0,255,200),3,cv2.LINE_AA)

    if move != "Still":
        d = {"Right ->":(50,0),"Left <-":(-50,0),"Up ^":(0,-50),"Down v":(0,50)}
        dx,dy = d.get(move,(0,0))
        cv2.arrowedLine(frame,(wx,wy),(wx+dx,wy+dy),mc,3,tipLength=0.35)

    draw_panel(frame,[
        ("Hand",    label,    (200,255,200)),
        ("Colour",  col_name, col_bgr),
        ("Fingers", fingers,  (100,255,255)),
        ("Move",    move,     mc),
        ("Position",pos,      (255,200,100)),
    ], x=10+idx*330, y=10)


# ── Legacy mode ───────────────────────────────────────────────────────────────
def run_legacy():
    mp_h   = mp.solutions.hands
    mp_dr  = mp.solutions.drawing_utils
    mp_st  = mp.solutions.drawing_styles
    det    = mp_h.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.7, min_tracking_confidence=0.6)
    cap    = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1)
        h,w   = frame.shape[:2]
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        res   = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        t1=time.time(); fps=1/max(t1-t0,1e-6); t0=t1
        cv2.putText(frame,f"FPS:{fps:.0f}",(w-100,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,255,180),2,cv2.LINE_AA)

        if res.multi_hand_landmarks:
            for idx,(hlms,hinfo) in enumerate(
                    zip(res.multi_hand_landmarks, res.multi_handedness)):
                label = hinfo.classification[0].label
                lms   = hlms.landmark        # legacy NormalizedLandmark list
                process_hand(frame, hsv, lms, label, idx)
        else:
            wrist_history.clear()
            cv2.putText(frame,"Camera munnaadi kai kaattungal",
                        (w//2-260,h//2),cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(100,100,255),2,cv2.LINE_AA)

        cv2.imshow("Hand Tracker", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows(); det.close()


# ── New Task API mode ─────────────────────────────────────────────────────────
def run_new_api():
    import urllib.request, os, tempfile

    model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("Model download பண்றோம்... (சிறிது நேரம் ஆகும்)")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("Download முடிந்தது!")

    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv

    opts = mpv.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=mpv.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    t0  = time.time()
    tms = 0

    with mpv.HandLandmarker.create_from_options(opts) as det:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            h,w   = frame.shape[:2]
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tms  += 33
            res   = det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), tms)

            t1=time.time(); fps=1/max(t1-t0,1e-6); t0=t1
            cv2.putText(frame,f"FPS:{fps:.0f}",(w-100,28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,255,180),2,cv2.LINE_AA)

            if res.hand_landmarks:
                for idx,(lms,hand) in enumerate(
                        zip(res.hand_landmarks, res.handedness)):
                    label = hand[0].display_name
                    process_hand(frame, hsv, lms, label, idx)
            else:
                wrist_history.clear()
                cv2.putText(frame,"Camera munnaadi kai kaattungal",
                            (w//2-260,h//2),cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,(100,100,255),2,cv2.LINE_AA)

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  Hand Tracker Starting...")
    print("  'q' press பண்ணினால் close ஆகும்")
    print("="*50)
    if USE_LEGACY:
        run_legacy()
    else:
        run_new_api()
    print("Stopped.")