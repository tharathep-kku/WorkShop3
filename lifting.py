"""โมเดลเชิงเรขาคณิตสำหรับประเมินงานยก — ย้ายมาจาก Workshop_LiftingWeightLimit_G13.ipynb

สูตรคำนวณ (calibrate · มุมหัน · มุมบิด · H/V · ตารางน้ำหนัก · ระยะเอื้อม) เหมือน notebook ทุกบรรทัด
ต่างกันแค่ค่าที่ปรับได้ย้ายจากตัวแปร global ไปอยู่ใน Params เพื่อให้ Streamlit ส่งค่าจาก slider เข้ามาได้
"""
import math, urllib.request, warnings
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings("ignore", category=RuntimeWarning)   # ไม่ต้องแสดงคำเตือนเมื่อเฉลี่ยค่าที่เป็น NaN ทั้งหมด (เหมือน notebook)

POSE_MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


@dataclass(frozen=True)
class Params:
    # ค่าที่ปรับได้ (ค่าตั้งต้น = ค่าที่กลุ่ม G13 ปรับไว้ ยกเว้น twist_warn_deg ที่ใช้ 45° ตามแบบประเมิน)
    calib_seconds: float = 1.5      # ช่วงยืนนิ่งต้นคลิป (วินาที)
    camera_fov_deg: float = 70.0    # มุมมองกล้องตามด้านยาวของภาพ (มือถือทั่วไป 65–75°)
    waist_fraction: float = 0.40    # ตำแหน่งเอวระหว่างสะโพกกับไหล่
    twist_warn_deg: float = 45.0    # เกณฑ์บิดตัวของแบบประเมิน (ขั้นตอนที่ 4)
    lift_min_move_cm: float = 20.0  # มือต้องเคลื่อนในแนวดิ่งอย่างน้อยเท่านี้ระหว่างหยิบกับวาง จึงนับ 1 รอบ


def ensure_pose_model():
    # ดาวน์โหลดโมเดล MediaPipe ครั้งแรกที่ใช้
    if not POSE_MODEL_PATH.is_file():
        POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)


def open_video(path):
    # เปิดวิดีโอ คืน (cap, ข้อมูลวิดีโอ)
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, info


def create_pose_detector():
    # สร้างตัวตรวจจับท่าทาง (สร้างครั้งเดียว ใช้ได้ทั้งคลิป)
    ensure_pose_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1)
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(detector, frame, frame_index, fps):
    # ② คืน xy (33, 2) พิกเซล · world (33, 3) เมตร · conf (33,)   (เป็น NaN ถ้าไม่พบคน)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(frame_index * 1000 / fps)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

    xy = np.full((33, 2), np.nan)
    world = np.full((33, 3), np.nan)
    conf = np.full(33, np.nan)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks[0]):
            xy[i] = [lm.x * w, lm.y * h]
            conf[i] = min(lm.visibility, lm.presence)
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            world[i] = [lm.x, lm.y, lm.z]
    return xy, world, conf


MIN_CONFIDENCE     = 0.5   # ค่าวัดที่ confidence ต่ำกว่านี้ไม่ใช้
MAX_PREDICT_FRAMES = 5     # ใช้ค่าทายแทนได้ติดกันกี่เฟรม
HISTORY_FRAMES     = 3     # moving average = เฟรมนี้ + 3 เฟรมก่อนหน้า
WORLD_WINDOW       = 5     # 👉 MODIFY — จำนวนเฟรมที่เฉลี่ยโครงร่าง 3 มิติ


def create_kalman(x, y):
    # Kalman filter แบบความเร็วคงที่  สถานะ = [x, y, vx, vy]  ค่าวัด = [x, y]
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    acc = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], np.float32)
    kf.processNoiseCov = 16.0 * acc @ acc.T                     # การเคลื่อนที่ไม่แน่นอนแค่ไหน
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 25.0  # ค่าวัดคลาดเคลื่อนประมาณ 5 px
    kf.errorCovPost = np.diag([25, 25, 400, 400]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


class PointTracker:
    # ติดตาม + ลดการสั่นของ keypoint 1 จุด

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.gate = max(25.0, 0.04 * math.hypot(width, height))   # ระยะกระโดดสูงสุดที่ยอมรับ (px)
        self.kalman = None       # None = ยังไม่ได้เริ่มติดตาม
        self.missing = 0         # จำนวนเฟรมติดกันที่ไม่มีค่าวัดที่ดี
        self.history = []        # ตำแหน่งล่าสุด สำหรับ moving average

    def update(self, x, y, conf):
        # คืน (ตำแหน่งที่กรองแล้ว, วัดได้จริงในเฟรมนี้หรือไม่)
        good = conf >= MIN_CONFIDENCE and 0 <= x < self.width and 0 <= y < self.height   # NaN → False
        point, observed = None, False

        if self.kalman is None:                          # ยังไม่ได้ติดตาม → เริ่มเมื่อค่าวัดดี
            if good:
                self.kalman = create_kalman(x, y)
                point, observed = np.array([x, y]), True
        else:
            predicted = self.kalman.predict()[:2, 0]     # (ก-1) predict
            jump = math.hypot(x - predicted[0], y - predicted[1]) if good else 0
            if good and jump <= self.gate:               # (ก-2) correct ด้วยค่าวัด
                self.kalman.correct(np.array([[x], [y]], np.float32))
                point, observed = np.array([x, y]), True
                self.missing = 0
            else:                                        # ไม่มีค่าวัดที่ดี → ใช้ค่าทาย
                self.missing += 1
                inside = 0 <= predicted[0] < self.width and 0 <= predicted[1] < self.height
                if self.missing <= MAX_PREDICT_FRAMES and inside:
                    point = predicted
                else:                                    # หายนานเกินไป → เริ่มใหม่
                    self.kalman, self.missing, self.history = None, 0, []

        if point is None:
            return np.array([np.nan, np.nan]), False
        self.history = (self.history + [point])[-(HISTORY_FRAMES + 1):]   # (ข) moving average
        return np.mean(self.history, axis=0), observed


class PoseSmoother:
    # PointTracker 33 ตัว (สำหรับภาพ) + ค่าเฉลี่ยย้อนหลังของโครงร่าง 3 มิติ

    def __init__(self, width, height):
        self.trackers = [PointTracker(width, height) for _ in range(33)]
        self.world_history = deque(maxlen=WORLD_WINDOW)

    def update(self, xy, world, conf):
        xy_s = np.full((33, 2), np.nan)
        observed = np.zeros(33, dtype=bool)
        for i in range(33):
            xy_s[i], observed[i] = self.trackers[i].update(xy[i, 0], xy[i, 1], conf[i])
        self.world_history.append(world)
        world_s = np.nanmean(np.array(self.world_history), axis=0)    # (ค)
        return xy_s, world_s, observed


SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24),                  # ลำตัว
            (11, 13), (13, 15), (15, 17), (15, 19),                  # แขนซ้าย
            (12, 14), (14, 16), (16, 18), (16, 20),                  # แขนขวา
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),        # ขาซ้าย
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),        # ขาขวา
            (0, 11), (0, 12)]                                        # ศีรษะ


def pt(p):
    # แปลงพิกัดทศนิยมเป็นจุดจำนวนเต็มสำหรับ OpenCV
    return (int(round(p[0])), int(round(p[1])))


def ui_scale(img):
    # ขนาดตัวอักษร/เส้น เทียบกับภาพกว้าง 720 px
    return img.shape[1] / 720


def draw_skeleton(img, xy_s, observed):
    # ④ วาดโครงกระดูกลงบนภาพ
    s = ui_scale(img)
    for a, b in SKELETON:
        if np.isfinite(xy_s[a]).all() and np.isfinite(xy_s[b]).all():
            cv2.line(img, pt(xy_s[a]), pt(xy_s[b]), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    for a, b in SKELETON:
        for p in (a, b):
            if np.isfinite(xy_s[p]).all():
                color = (0, 200, 0) if observed[p] else (0, 140, 255)
                cv2.circle(img, pt(xy_s[p]), max(2, int(4 * s)), color, -1, cv2.LINE_AA)
    return img


NOSE_ABOVE_FOOT = 0.905     # พื้น→จมูก / ส่วนสูง
FEET = [29, 30, 31, 32]     # ส้นเท้า + ปลายเท้า (ใช้หาระดับพื้น)
LOWER = [27, 28, 29, 30, 31, 32]   # ข้อเท้า + ส้นเท้า + ปลายเท้า (ใช้หาแกนลำตัว เท้า→สะโพก)
PNP_POINTS = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def unit(v):
    # ทำเวกเตอร์ให้ยาว 1
    return v / np.linalg.norm(v)


def mean_point(points, ids):
    # จุดเฉลี่ยของหลาย landmark (ข้ามค่า NaN)
    return np.nanmean(points[ids], axis=0)


# ---------------------------------------------------------------- 5.1
def calc_cm_per_px(xy_frames, height_cm):
    # ซม. ต่อพิกเซล จากระยะพื้น→จมูกในภาพ
    distances = []
    for xy in xy_frames:
        floor_y = np.nanmax(xy[FEET, 1])
        distances.append(floor_y - xy[0, 1])
    return height_cm * NOSE_ABOVE_FOOT / np.nanmedian(distances)


def camera_matrix(width, height, fov_deg):
    # เมทริกซ์กล้องรูเข็ม K จาก FOV
    f = (max(width, height) / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


# ---------------------------------------------------------------- 5.2
def calc_world_scale(world_frames, height_cm):
    # ตัวคูณที่ทำให้โครงร่าง 3 มิติสูง (พื้น→จมูก) เท่าผู้ยกจริง
    heights = []
    for world in world_frames:
        axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))  # เท้า→สะโพก
        heights.append(np.nanmax((world[0] - world[FEET]) @ axis) * 100)    # พื้น→จมูก (ซม.)
    return height_cm * NOSE_ABOVE_FOOT / np.nanmedian(heights)


def solve_camera_pose(P, xy, observed, K, previous=None):
    # หา R, t ที่ทำให้จุด 3 มิติ P ฉายลงภาพแล้วตรงกับ xy มากที่สุด
    ids = [i for i in PNP_POINTS if observed[i]]
    if len(ids) < 6:
        return None
    ok = False
    if previous is not None:                                 # ใช้ผลเฟรมก่อนเป็นจุดเริ่ม → นิ่งและเร็ว
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None,
                                      previous["rvec"].copy(), previous["tvec"].copy(), True)
    # แอป: solvePnP บางครั้งคืนคำตอบกระจกเงาที่คนอยู่หลังกล้อง (tz < 0) แล้วความสูงมือเพี้ยนทั้งคลิป
    # → ถ้าเจอ ให้แก้ใหม่โดยไม่ใช้จุดเริ่ม แล้วลองอีกวิธี ถ้ายังผิดถือว่าเฟรมนี้วัดไม่ได้
    for flags in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP):
        if ok and tvec[2, 0] > 0:
            break
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None, flags=flags)
    if not ok or tvec[2, 0] <= 0:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "rvec": rvec, "tvec": tvec}


def to_camera_3d(xy, P, pose, K):
    # ตำแหน่งในภาพ (u, v) + ความลึก Z จากโครงร่าง → พิกัดกล้อง 3 มิติ (33, 3) ซม.
    P_cam = P @ pose["R"].T + pose["t"]                      # โครงร่างในพิกัดกล้อง
    Z = P_cam[:, 2]
    X = (xy[:, 0] - K[0, 2]) * Z / K[0, 0]
    Y = (xy[:, 1] - K[1, 2]) * Z / K[1, 1]
    C = np.column_stack([X, Y, Z])
    missing = np.isnan(xy[:, 0])
    C[missing] = P_cam[missing]                              # จุดที่ไม่เห็นในภาพ → ใช้โครงร่างแทน
    return C


def project(points, K):
    # จุดในพิกัดกล้อง (…, 3) → ตำแหน่งในภาพ (…, 2) พิกเซล
    points = np.asarray(points, dtype=float)
    u = K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2]
    v = K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------- 5.3
def body_frame(C, up, floor=None):
    # กรอบพิกัดร่างกาย: O, up, side, front  (floor = จุดบนพื้นจาก calibrate)
    ankle = mean_point(C, [27, 28])
    side = C[24] - C[23]
    side = unit(side - (side @ up) * up)                     # เส้นสะโพกในแนวราบ
    front = unit(np.cross(side, up))                         # ตั้งฉากกับระนาบสะโพก
    toe, heel = mean_point(C, [31, 32]), mean_point(C, [29, 30])
    if (toe - heel) @ front < 0:                             # ให้ front ชี้ไปทางปลายเท้า
        front = -front
    if floor is None:
        drop = np.nanmin((C[FEET] - ankle) @ up)             # พื้น = จุดเท้าที่ต่ำที่สุด
    else:
        drop = (floor - ankle) @ up
    O = ankle + up * drop                                    # กึ่งกลางข้อเท้า ที่ระดับพื้น
    if not (np.isfinite(O).all() and np.isfinite(front).all()):
        return None
    return {"O": O, "up": up, "side": side, "front": front}


def body_coords(p, body):
    # จุด 3 มิติ → (ด้านหน้า, ความสูง, ด้านข้าง) ซม. เทียบกับ O
    d = p - body["O"]
    return d @ body["front"], d @ body["up"], d @ body["side"]


def twist_angle(world):
    # มุม (มีเครื่องหมาย) ระหว่างเส้นไหล่กับเส้นสะโพก รอบแกนลำตัว — ใช้ในขั้นที่ ⑦
    axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))
    hip = world[24] - world[23]
    shoulder = world[12] - world[11]
    hip = unit(hip - (hip @ axis) * axis)
    shoulder = unit(shoulder - (shoulder @ axis) * axis)
    return math.degrees(math.atan2(np.cross(hip, shoulder) @ axis, hip @ shoulder))


# ---------------------------------------------------------------- รวมทุกอย่าง
def calibrate(samples, height_cm, width, height, p):
    # ⑤ samples = รายการ {"xy", "world", "observed"} ของช่วงยืนนิ่ง → dict calib
    calib = {"height_cm": height_cm}
    calib["cm_per_px"] = calc_cm_per_px([s["xy"] for s in samples], height_cm)               # 5.1
    calib["K"] = camera_matrix(width, height, p.camera_fov_deg)
    calib["camera_dist_cm"] = calib["K"][0, 0] * calib["cm_per_px"]
    calib["world_scale"] = calc_world_scale([s["world"] for s in samples], height_cm)          # 5.2

    C_list, axes, pose = [], [], None                                                        # พิกัด 3 มิติของแต่ละเฟรม
    for s in samples:
        P = s["world"] * 100 * calib["world_scale"]
        pose = solve_camera_pose(P, s["xy"], s["observed"], calib["K"], pose)
        if pose is None:
            continue
        C = to_camera_3d(s["xy"], P, pose, calib["K"])
        C_list.append(C)
        axes.append(unit(mean_point(C, [23, 24]) - mean_point(C, LOWER)))
    calib["up"] = unit(np.mean(axes, axis=0))                                                # 5.3 แนวดิ่ง
    floors = []                                                                              # จุดบนพื้น
    for C in C_list:
        body = body_frame(C, calib["up"])
        if body is not None:
            floors.append(body["O"])
    calib["floor"] = np.median(floors, axis=0)

    knee, hip, shoulder = [], [], []                                                         # ระดับเข่า/สะโพก/ไหล่
    for C in C_list:
        body = body_frame(C, calib["up"], calib["floor"])
        if body is None:
            continue
        knee.append(body_coords(mean_point(C, [25, 26]), body)[1])
        hip.append(body_coords(mean_point(C, [23, 24]), body)[1])
        shoulder.append(body_coords(mean_point(C, [11, 12]), body)[1])
    calib["knee_cm"] = np.median(knee)
    calib["shoulder_cm"] = np.median(shoulder)
    calib["waist_cm"] = np.median(hip) + p.waist_fraction * (calib["shoulder_cm"] - np.median(hip))
    calib["twist_baseline"] = np.nanmedian([twist_angle(s["world"]) for s in samples])       # ใช้ในขั้นที่ ⑦
    calib["pose"] = pose                                     # แอป: ใช้เป็นจุดเริ่มของเฟรมถัดไป
    return calib


def orientation_angle(C):
    # ⑥ มุมระหว่างเส้นสะโพกกับระนาบกล้อง (0 = หันเข้ากล้อง, 90 = หันข้าง)
    d = C[24] - C[23]
    yaw = abs(math.degrees(math.atan2(d[2], d[0])))    # 0…180
    return min(yaw, 180 - yaw)                         # พับเป็น 0…90


def twist_from_baseline(world, calib):
    # ⑦ มุมบิดตัว (องศา, ค่าบวก) เทียบกับท่ายืนตรง
    diff = twist_angle(world) - calib["twist_baseline"]
    diff = (diff + 180) % 360 - 180                    # ให้อยู่ในช่วง −180…180
    return abs(diff)


REACH_LIMITS_CM = [7 * 2.54, 12 * 2.54]                        # 17.78, 30.48 ซม.
ROW_NAMES = ["Above Shoulder", "Waist-Shoulder", "Knee-Waist", "Below Knee"]
COL_NAMES = ["Near", "Mild", "Extended"]
WEIGHT_LIMITS_KG = [[29, 18, 14],      # Above Shoulder
                    [32, 23, 18],      # Waist to Shoulder
                    [41, 25, 18],      # Knee to Waist  (สูงสุด)
                    [31, 23, 16]]      # Below Knee
LEFT_HAND, RIGHT_HAND = [15, 17, 19], [16, 18, 20]


def table_row(v_cm, calib):
    # แถวของตารางจากความสูงมือ
    if v_cm >= calib["shoulder_cm"]:
        return 0
    if v_cm >= calib["waist_cm"]:
        return 1
    if v_cm >= calib["knee_cm"]:
        return 2
    return 3


def table_col(h_cm):
    # คอลัมน์ของตารางจากระยะ H
    if h_cm < REACH_LIMITS_CM[0]:
        return 0
    if h_cm < REACH_LIMITS_CM[1]:
        return 1
    return 2


def measure_hands(C, xy, calib):
    # ⑧-1 คืน H, V, ช่องในตาราง และน้ำหนักสูงสุด (None ถ้าวัดไม่ได้)
    body = body_frame(C, calib["up"], calib["floor"])
    if body is None:
        return None
    hands_3d = np.nanmean([mean_point(C, LEFT_HAND), mean_point(C, RIGHT_HAND)], axis=0)
    hands_px = np.nanmean([mean_point(xy, LEFT_HAND), mean_point(xy, RIGHT_HAND)], axis=0)
    if np.isnan(hands_3d).any():
        return None
    if np.isnan(hands_px).any():                            # มือหลุดจากภาพ → ฉายจาก 3 มิติแทน
        hands_px = project(hands_3d, calib["K"])

    front, v_cm, side = body_coords(hands_3d, body)
    h_cm = math.hypot(front, side)                          # NIOSH H
    row, col = table_row(v_cm, calib), table_col(h_cm)
    return {"H": h_cm, "V": v_cm, "front": front, "side": side,
            "row": row, "col": col, "limit_kg": WEIGHT_LIMITS_KG[row][col],
            "hands_px": hands_px, "body": body,
            "direction": -1 if front < -REACH_LIMITS_CM[0] else 1}   # มือหลังข้อเท้าชัดเจน → ตารางยื่นไปด้านหลัง


TABLE_LENGTH_CM = 55.0     # 👉 MODIFY — ความยาวตารางด้านหน้า (ซม.)
ABOVE_MARGIN_CM = 30.0     # ความสูงของแถว Above Shoulder เหนือไหล่
HEADER_CM       = 10.0     # แถบหัวคอลัมน์
PX_PER_CM       = 8        # ความละเอียดของภาพตาราง
CELL_ALPHA      = 0.45     # 👉 MODIFY — ความทึบของช่อง


def cell_color(kg):
    # สีเดียวไล่ความเข้ม: น้ำหนักมาก (ปลอดภัยกว่า) = ครีมอ่อน · น้อย = แดงเข้ม
    t = (41 - kg) / (41 - 14)
    light, dark = np.array([160, 225, 255]), np.array([20, 20, 170])
    return tuple(int(c) for c in light + (dark - light) * t)


def make_table_image(calib, highlight, axis_on_left):
    # ⑧-2 วาดตารางแบน: คืน (ภาพสี, ภาพความทึบ 0–1)
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    w, h = int(TABLE_LENGTH_CM * PX_PER_CM), int(top_cm * PX_PER_CM)
    color = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.float32)

    col_edges = [0, REACH_LIMITS_CM[0], REACH_LIMITS_CM[1], TABLE_LENGTH_CM]            # ซม. จากลำตัว
    row_edges = [calib["shoulder_cm"] + ABOVE_MARGIN_CM, calib["shoulder_cm"],
                 calib["waist_cm"], calib["knee_cm"], 0]                               # ซม. จากพื้น (บน→ล่าง)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def to_x(cm):                     # ซม. จากลำตัว → x บนภาพตาราง
        x = int(cm * PX_PER_CM)
        return x if axis_on_left else w - x

    def to_y(cm):                     # ซม. จากพื้น → y บนภาพตาราง
        return int((top_cm - cm) * PX_PER_CM)

    def write(text, pos, size, thick, rgb=(255, 255, 255)):
        cv2.putText(color, text, pos, font, size, rgb, thick, cv2.LINE_AA)
        mask = np.zeros((h, w), np.uint8)                  # G13: OpenCV 5 วาดตัวอักษรได้เฉพาะภาพ uint8
        cv2.putText(mask, text, pos, font, size, 255, thick, cv2.LINE_AA)
        np.maximum(alpha, mask / 255.0, out=alpha, casting="unsafe")

    for r in range(4):
        for c in range(3):
            x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
            y0, y1 = to_y(row_edges[r]), to_y(row_edges[r + 1])
            kg = WEIGHT_LIMITS_KG[r][c]
            is_hand_cell = (r, c) == highlight
            cv2.rectangle(color, (x0, y0), (x1, y1), (0, 230, 255) if is_hand_cell else cell_color(kg), -1)
            alpha[y0:y1, x0:x1] = 0.85 if is_hand_cell else CELL_ALPHA
            cv2.rectangle(color, (x0, y0), (x1, y1), (255, 255, 255), 2)
            border = np.zeros((h, w), np.uint8)                # G13: วาดขอบบนภาพ uint8 แล้วค่อยใส่ใน alpha
            cv2.rectangle(border, (x0, y0), (x1, y1), 255, 2)
            alpha[border > 0] = 1.0
            write(str(kg), ((x0 + x1) // 2 - 22, (y0 + y1) // 2 + 15), 1.6, 4,
                  (0, 0, 0) if is_hand_cell else (255, 255, 255))
        x_label = 8 if not axis_on_left else w - 190
        write(ROW_NAMES[r], (x_label, to_y(row_edges[r]) + 26), 0.7, 2)
    for c in range(3):
        x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
        write(COL_NAMES[c], (x0 + 8, to_y(row_edges[0]) - 14), 0.7, 2)
    write("MAX WEIGHT (kg)", (w // 2 - 120, 30), 0.8, 2)
    return color, alpha


def draw_weight_table(img, calib, m):
    # ⑧-3 ฉายมุมตาราง 3 มิติลงภาพ แล้ว warpPerspective ภาพตารางลงไป
    body, K = m["body"], calib["K"]
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    corners_3d = np.array([O,                                               # ล่าง-ลำตัว
                           O + front * TABLE_LENGTH_CM,                     # ล่าง-ปลาย
                           O + front * TABLE_LENGTH_CM + up * top_cm,       # บน-ปลาย
                           O + up * top_cm])                                # บน-ลำตัว
    if (corners_3d[:, 2] < 10).any():                                       # อยู่หลังกล้อง → ไม่วาด
        return img
    corners_px = project(corners_3d, K).astype(np.float32)
    if abs(cv2.contourArea(corners_px)) < 0.002 * img.shape[0] * img.shape[1]:
        return img                                                          # แผ่นหันสันเข้ากล้อง → ไม่วาด

    axis_on_left = corners_px[1, 0] >= corners_px[0, 0]                     # ตารางยื่นไปทางขวาของภาพ?
    key = (m["row"], m["col"], axis_on_left)                                # แอป: ภาพตารางมีได้แค่ 24 แบบต่อคลิป → วาดครั้งเดียวแล้วเก็บ
    cache = calib.setdefault("table_cache", {})
    if key not in cache:
        cache[key] = make_table_image(calib, key[:2], axis_on_left)
    table, alpha = cache[key]
    th, tw = alpha.shape
    x_body, x_end = (0, tw) if axis_on_left else (tw, 0)
    table_corners = np.float32([[x_body, th], [x_end, th], [x_end, 0], [x_body, 0]])

    M = cv2.getPerspectiveTransform(table_corners, corners_px)              # เมทริกซ์ 3×3
    size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(table, M, size)
    warped_alpha = cv2.warpPerspective(alpha, M, size)[:, :, None]
    img[:] = (img * (1 - warped_alpha) + warped * warped_alpha).astype(np.uint8)
    return img


def draw_hand_label(img, calib, m):
    # ⑧-4 จุด (H, V) บนตาราง + ป้ายน้ำหนักสูงสุด
    body, K, s = m["body"], calib["K"], ui_scale(img)
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    on_table = project(O + front * m["H"] + up * m["V"], K)                 # (H, V) บนแผ่นตาราง
    hands = m["hands_px"]
    cv2.line(img, pt(hands), pt(on_table), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.circle(img, pt(on_table), int(7 * s), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt(hands), int(16 * s), (0, 230, 255), max(2, int(3 * s)), cv2.LINE_AA)

    # ป้ายวางที่ขอบปลายตาราง ระดับเดียวกับมือ
    anchor = project(O + front * (TABLE_LENGTH_CM + 3) + up * m["V"], K)
    box_w, box_h = int(230 * s), int(80 * s)
    x0 = anchor[0] if anchor[0] >= project(O, K)[0] else anchor[0] - box_w
    x0 = int(np.clip(x0, 5, img.shape[1] - box_w - 5))
    y0 = int(np.clip(anchor[1] - box_h / 2, 5, img.shape[0] - box_h - 5))
    cv2.line(img, pt(on_table), (x0 + box_w // 2, y0 + box_h // 2), (0, 230, 255), max(2, int(3 * s)))
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 230, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"MAX {m['limit_kg']} kg", (x0 + int(10 * s), y0 + int(35 * s)), font, 1.0 * s,
                (0, 0, 0), max(2, int(3 * s)), cv2.LINE_AA)
    cv2.putText(img, f"{ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}", (x0 + int(10 * s), y0 + int(55 * s)),
                font, 0.45 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, f"H {m['H']:.0f} cm   V {m['V']:.0f} cm", (x0 + int(10 * s), y0 + int(72 * s)),
                font, 0.45 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
    return img


REACH_WARN_RATIO = 0.6   # 👉 MODIFY — มือห่างแนวลำตัวเกินกี่เท่าของความยาวลำตัว ถือว่าเอื้อมไกล


def point_to_segment(p, a, b):
    # ระยะสั้นที่สุดจากจุด p ถึงส่วนของเส้นตรง ab (จำกัด t ให้อยู่ระหว่าง 0–1)
    ab, ap = b - a, p - a
    denom = float(ab @ ab)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    t = float(np.clip((ap @ ab) / denom, 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def reach_distance(C):
    # ⑧.5 ระยะเอื้อมของกลุ่ม G13 (ซม.) ข้างขวา · ข้างซ้าย · ข้างที่ไกลกว่า · สัดส่วนต่อความยาวลำตัว
    right = point_to_segment((C[16] + C[20]) / 2, C[12], C[24])
    left = point_to_segment((C[15] + C[19]) / 2, C[11], C[23])
    torso = np.nanmean([np.linalg.norm(C[12] - C[24]), np.linalg.norm(C[11] - C[23])])
    sides = [d for d in (right, left) if np.isfinite(d)]
    best = max(sides) if sides else np.nan
    ratio = best / torso if np.isfinite(best) and torso > 0 else np.nan
    return {"reach_r_cm": right, "reach_l_cm": left, "reach_cm": best, "reach_ratio": ratio}


def draw_info_panel(img, lines):
    # แผงข้อความมุมซ้ายบน: lines = [(ข้อความ, สี), ...]
    s = ui_scale(img)
    line_h = int(26 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (int(360 * s), int(15 * s) + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (int(10 * s), int(28 * s) + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s, color, max(1, int(1.5 * s)), cv2.LINE_AA)
    return img


def draw_lift_counter(img, count, lifting, flash):
    # G13: กล่องตัวนับรอบมุมขวาบน (เขียวกะพริบเมื่อเพิ่งนับรอบใหม่ · LIFT = กำลังถือของระหว่างหยิบกับวาง)
    s = ui_scale(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_w, box_h = int(170 * s), int(72 * s)
    x0, y0 = img.shape[1] - box_w - int(10 * s), int(10 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (40, 150, 40) if flash else (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    cv2.putText(img, "REP", (x0 + int(10 * s), y0 + int(26 * s)), font, 0.6 * s, (255, 255, 255),
                max(1, int(1.5 * s)), cv2.LINE_AA)
    if lifting:
        cv2.putText(img, "LIFT", (x0 + int(10 * s), y0 + int(58 * s)), font, 0.6 * s, (80, 255, 80),
                    max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, str(count), (x0 + int(95 * s), y0 + int(60 * s)), font, 1.6 * s, (0, 230, 255),
                max(2, int(3 * s)), cv2.LINE_AA)
    return img


def extract_video_keypoints(video_path, progress=None):
    # G13: รัน MediaPipe ทั้งคลิปครั้งเดียว → ปรับค่าแล้วคำนวณใหม่ได้เร็วโดยไม่ต้องรัน MediaPipe ซ้ำ
    # progress(สัดส่วน 0–1) ถูกเรียกเป็นระยะ ใช้แสดงแถบความคืบหน้า
    cap, info = open_video(video_path)
    xys, worlds, confs = [], [], []
    with create_pose_detector() as detector:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            xy, world, conf = extract_keypoints(detector, frame, i, info["fps"])
            xys.append(xy); worlds.append(world); confs.append(conf)
            i += 1
            if progress and i % 10 == 0:
                progress(min(1.0, i / max(1, info["frames"])))
    cap.release()
    return {"xy": np.array(xys), "world": np.array(worlds), "conf": np.array(confs)}


RENDER_MAX_PX = 960    # ด้านยาวของวิดีโอผลลัพธ์ · ย่อเฟรมก่อนวาด เพราะวาดตารางบนภาพ 1080p ช้าเกินไปสำหรับ Streamlit Cloud


def run_pipeline(video_path, out_path, height_cm, p, keypoints=None, progress=None, lifts=()):
    # out_path = None → ไม่เขียนวิดีโอ (เร็ว ใช้ตอนปรับค่า)
    # lifts = ผลจาก detect_lifts() ของรอบคำนวณก่อนหน้า → ใช้วาดตัวนับรอบบนวิดีโอ
    # keypoints = ผลจาก extract_video_keypoints() → ไม่ต้องรัน MediaPipe ซ้ำ
    cap, info = open_video(video_path)
    write = out_path is not None
    scale = min(1.0, RENDER_MAX_PX / max(info["width"], info["height"]))
    out_size = (int(info["width"] * scale) // 2 * 2, int(info["height"] * scale) // 2 * 2)   # H.264 ต้องเป็นเลขคู่
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], out_size) if write else None
    smoother = PoseSmoother(info["width"], info["height"])
    n_calib = int(round(p.calib_seconds * info["fps"]))
    calib, samples, pose = None, [], None
    records = []
    WHITE, YELLOW, RED = (255, 255, 255), (0, 230, 255), (80, 80, 255)
    n_cached = len(keypoints["xy"]) if keypoints is not None else None

    with (create_pose_detector() if keypoints is None else nullcontext()) as detector:
        i = 0
        while True:
            if keypoints is not None and i >= n_cached:
                break
            frame = None
            if write or keypoints is None:
                ok, frame = cap.read()                                         # ① อ่านเฟรม
                if not ok:
                    break
            if keypoints is None:
                xy, world, conf = extract_keypoints(detector, frame, i, info["fps"])   # ② keypoint
            else:
                xy, world, conf = keypoints["xy"][i], keypoints["world"][i], keypoints["conf"][i]
            xy_s, world_s, observed = smoother.update(xy, world, conf)         # ③ ลดการสั่น
            img = cv2.resize(frame, out_size) if write else None
            xy_d = xy_s * scale                                                # พิกัดบนภาพย่อ ใช้เฉพาะตอนวาด
            row = {"frame_number": i + 1, "time_s": (i + 1) / info["fps"]}

            if calib is None:                                                  # ⑤ ช่วงยืนนิ่ง
                samples.append({"xy": xy_s, "world": world_s, "observed": observed})
                if len(samples) == n_calib:
                    calib = calibrate(samples, height_cm, info["width"], info["height"], p)
                    pose = calib["pose"]
                    K_d = calib["K"].copy()
                    K_d[:2] *= scale
                    calib_d = {**calib, "K": K_d}                              # calib สำหรับวาดบนภาพย่อ
                if write:
                    img = draw_skeleton(img, xy_d, observed)                   # ④
                    img = draw_info_panel(img, [(f"CALIBRATING... stand still {len(samples)}/{n_calib}", YELLOW)])
            else:
                P = world_s * 100 * calib["world_scale"]                       # ⑤ พิกัด 3 มิติ
                pose = solve_camera_pose(P, xy_s, observed, calib["K"], pose)
                m = None
                if pose is not None:
                    C = to_camera_3d(xy_s, P, pose, calib["K"])
                    row["orientation_deg"] = orientation_angle(C)              # ⑥
                    m = measure_hands(C, xy_s, calib)                          # ⑧ H, V
                    row.update(reach_distance(C))                              # G13 ⑧.5 ระยะเอื้อม
                row["twist_deg"] = twist_from_baseline(world_s, calib)         # ⑦

                if m is not None:
                    row.update({"H_cm": m["H"], "V_cm": m["V"], "zone": ROW_NAMES[m["row"]],
                                "reach_class": COL_NAMES[m["col"]], "limit_kg": m["limit_kg"]})
                if write:
                    if m is not None:
                        m = {**m, "hands_px": m["hands_px"] * scale}
                        img = draw_weight_table(img, calib_d, m)               # ⑧ ตาราง
                    img = draw_skeleton(img, xy_d, observed)                   # ④
                    reach_cm = row.get("reach_cm", np.nan)
                    lines = [(f"scale   {calib['cm_per_px']:.3f} cm/px", WHITE),
                             (f"orient  {row.get('orientation_deg', np.nan):.0f} deg to camera", WHITE),
                             (f"twist   {row['twist_deg']:.0f} deg", RED if row["twist_deg"] >= p.twist_warn_deg else WHITE),
                             (f"reach   {reach_cm:.0f} cm ({row.get('reach_ratio', np.nan):.2f} x torso)",
                              RED if row.get("reach_ratio", 0) > REACH_WARN_RATIO else WHITE)]
                    if m is not None:
                        img = draw_hand_label(img, calib_d, m)                 # ⑧ ป้ายน้ำหนัก
                        lines += [(f"hands   H {m['H']:.0f} cm | V {m['V']:.0f} cm", WHITE),
                                  (f"zone    {ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}", WHITE),
                                  (f"LIMIT   {m['limit_kg']} kg", YELLOW)]
                    img = draw_info_panel(img, lines)
                    done = [l["place_frame"] for l in lifts if l["place_frame"] <= i + 1]
                    img = draw_lift_counter(img, len(done),                    # G13 ตัวนับรอบ
                                            any(l["pick_frame"] <= i + 1 < l["place_frame"] for l in lifts),
                                            bool(done) and i + 1 - max(done) <= info["fps"] * 0.5)

            if write:
                writer.write(img)                                              # ⑨ เขียนวิดีโอ
            records.append(row)
            i += 1
            if progress and i % 10 == 0:
                progress(min(1.0, i / max(1, n_cached or info["frames"])))
    cap.release()
    if write:
        writer.release()
    return pd.DataFrame(records), calib


# ---------------------------------------------------------------- นับรอบยก + จังหวะหยิบ/วาง
# แทน LiftCounter ของ notebook ที่นับเฉพาะจังหวะมือขึ้น ≥ 35 ซม. ซึ่งพลาดงานยกลงและงานยกระยะสั้น
REST_TOL_CM  = 8.0    # V และ H ห่างจากท่ายืนพัก (มือห้อยข้างตัว) ไม่เกินเท่านี้ ถือว่าเป็นท่าพัก
REST_MIN_SEC = 0.4    # ต้องอยู่ท่าพักต่อเนื่องอย่างน้อยเท่านี้ จึงถือว่าจบรอบ
SMOOTH_SEC   = 0.5    # median ของ V, H ก่อนตัดสิน (ตัดค่ากระโดดเฟรมเดียว)


def detect_lifts(results, calib, fps, p):
    # 1 รอบยก = 1 ช่วงที่มือออกจากท่าพักแล้วกลับมาพัก โดยมือเคลื่อนแนวดิ่ง ≥ lift_min_move_cm
    # ในช่วงนั้น จุดที่มือต่ำสุดกับสูงสุดคือจังหวะหยิบและวาง (อันไหนมาก่อน = หยิบ) จึงใช้ได้ทั้งยกขึ้นและยกลง
    # ponytail: ต้องกลับมายืนพักมือระหว่างรอบ ถ้ายกต่อเนื่องไม่พักจะนับรวมเป็นรอบเดียว
    #           แก้โดยแบ่งช่วงด้วยจุดกลับทิศของ V เมื่อต้องรองรับงานยกต่อเนื่อง
    s = results.dropna(subset=["V_cm", "H_cm"]) if "V_cm" in results else results.iloc[:0]
    if s.empty:
        return []
    win = max(3, int(SMOOTH_SEC * fps)) | 1
    V = s["V_cm"].rolling(win, center=True, min_periods=1).median()
    H = s["H_cm"].rolling(win, center=True, min_periods=1).median()
    twist = s["twist_deg"].rolling(win, center=True, min_periods=1).median()
    first = max(1, int(0.5 * fps))                               # ท่าพัก = ครึ่งวินาทีแรกหลัง calibrate
    rest_V, rest_H = V.iloc[:first].median(), H.iloc[:first].median()
    rest = ((V - rest_V).abs() < REST_TOL_CM) & ((H - rest_H).abs() < REST_TOL_CM)
    runs = (rest != rest.shift()).cumsum()
    rest &= rest.groupby(runs).transform("size") >= REST_MIN_SEC * fps   # พักแวบเดียวไม่นับ
    runs = (rest != rest.shift()).cumsum()

    half = max(1, int(0.25 * fps))

    def station(idx):
        # ตำแหน่งมือ ณ จังหวะหยิบ/วาง → ช่องในตารางน้ำหนัก (ขั้นตอนที่ 2 ของแบบประเมิน)
        k = s.index.get_loc(idx)
        v, h = V.iloc[max(0, k - half):k + half + 1].median(), H.iloc[max(0, k - half):k + half + 1].median()
        row, col = table_row(v, calib), table_col(h)
        return {"frame": int(s.loc[idx, "frame_number"]), "time_s": float(s.loc[idx, "time_s"]), "V": float(v),
                "H": float(h), "zone": ROW_NAMES[row], "reach_class": COL_NAMES[col],
                "limit_kg": WEIGHT_LIMITS_KG[row][col]}

    lifts = []
    for _, part in s[~rest].groupby(runs[~rest]):
        v = V[part.index]
        if v.max() - v.min() < p.lift_min_move_cm:
            continue
        a, b = sorted([v.idxmin(), v.idxmax()])
        pick, place = station(a), station(b)
        lifts.append({"lift": len(lifts) + 1, "pick": pick, "place": place,
                      "pick_frame": pick["frame"], "place_frame": place["frame"],
                      "limit_kg": min(pick["limit_kg"], place["limit_kg"]),      # ค่าที่เข้มกว่าระหว่างหยิบกับวาง
                      "twist_max_deg": float(twist.loc[a:b].max()),              # บิดตัวสูงสุดขณะถือของ
                      "reach_max_cm": float(s.loc[a:b, "reach_cm"].max())})
    return lifts


# ---------------------------------------------------------------- แบบประเมินความเสี่ยงงานยกเคลื่อนย้าย
# ที่มา: จันจิราภรณ์ วิชัย, สุนิสา ชายเกลี้ยง. วารสารวิจัย มข. 2557; 19(5): 708-719.
FREQ_ROWS = ["ยกน้อยกว่า 1 ครั้งทุก 5 นาที", "ยก 1 ครั้งทุก 2-5 นาที", "ยก 1 ครั้งทุก 1 นาที", "ยก 2-3 ครั้งทุก 1 นาที", "ยก 4-5 ครั้งทุก 1 นาที",
             "ยก 6-7 ครั้งทุก 1 นาที", "ยก 8-9 ครั้งทุก 1 นาที", "ยกมากกว่า 10 ครั้งทุก 1 นาที"]
HOUR_COLS = ["น้อยกว่า 1 ชม.", "1-2 ชม.", "มากกว่า 2 ชม."]
FREQ_MULTIPLIER = [[1.0, 1.0, 1.0],                             # หมายเหตุใต้ตาราง
                   [1.0, 0.95, 0.85],
                   [0.95, 0.9, 0.75],
                   [0.9, 0.85, 0.65],
                   [0.85, 0.7, 0.45],
                   [0.75, 0.5, 0.25],
                   [0.6, 0.35, 0.15],
                   [0.3, 0.2, 0.0]]
TWIST_MULTIPLIER = 0.85                                          # บิดลำตัว ≥ เกณฑ์ (45° ตามแบบประเมิน)
RISK_LEVELS = [(1, "ภาวะที่ยอมรับได้"),                            # LH Index < 50
               (2, "งานนั้นควรมีการตรวจสอบและติดตาม"),               # 50–75
               (3, "งานนั้นเริ่มมีปัญหาควรตรวจสอบเพื่อปรับปรุง"),        # > 75–100
               (4, "งานนั้นมีปัญหาควรแก้ไขปรับปรุงโดยทันที")]          # > 100


def freq_row(lifts_per_min):
    # ขั้นตอนที่ 3: แถวของตารางความถี่ · ค่าที่ตกระหว่างแถว (เช่น 1 ครั้งทุก 1.5 นาที · 10 ครั้ง/นาที) ใช้แถวที่เข้มกว่า
    if lifts_per_min < 0.2:
        return 0
    if lifts_per_min <= 0.5:
        return 1
    if lifts_per_min < 1.5:
        return 2
    return min(7, round(lifts_per_min) // 2 + 2)


def assess(weight_kg, limit_kg, freq_multiplier, twist_deg, twist_warn_deg):
    # ขั้นตอนที่ 4–7: ตัวคูณบิดตัว → RWL → LH Index → ระดับความเสี่ยง
    twist_mult = TWIST_MULTIPLIER if twist_deg >= twist_warn_deg else 1.0
    rwl = limit_kg * freq_multiplier * twist_mult                # ขั้นตอนที่ 5
    index = weight_kg / rwl * 100 if rwl > 0 else math.inf       # ขั้นตอนที่ 6
    level, meaning = RISK_LEVELS[(index >= 50) + (index > 75) + (index > 100)]   # ขั้นตอนที่ 7
    return {"twist_multiplier": twist_mult, "rwl_kg": rwl, "lh_index": index, "level": level, "meaning": meaning}


def demo():
    # ตรวจตัวเลขกับแบบประเมิน: ตารางความถี่ · RWL · LH Index · ขอบระดับความเสี่ยง
    assert [freq_row(r) for r in (0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)] == [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    assert len(FREQ_ROWS) == len(FREQ_MULTIPLIER) == 8 and FREQ_MULTIPLIER[5] == [0.75, 0.5, 0.25]
    a = assess(10, 25, 0.75, 50, 45)                             # 25 × 0.75 × 0.85 = 15.9375 กก.
    assert abs(a["rwl_kg"] - 15.9375) < 1e-9 and abs(a["lh_index"] - 62.745) < 1e-3 and a["level"] == 2
    assert [assess(w, 20, 1.0, 0, 45)["level"] for w in (9.9, 10, 15, 15.1, 20, 20.1)] == [1, 2, 2, 3, 3, 4]
    assert assess(5, 20, 0.0, 0, 45)["level"] == 4               # ตัวคูณความถี่ 0 → ยกไม่ได้เลย
    print("lifting.py: ตัวเลขตรงกับแบบประเมิน")


if __name__ == "__main__":
    demo()
