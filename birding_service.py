#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, time, threading, os, traceback, json, re
import numpy as np
from pathlib import Path
from datetime import datetime, time as dtime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ======================
# 配置（支持环境变量覆盖；摄像头凭据按需求保留硬编码默认值）
# ======================
def _env(name, default, cast=str):
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


# 摄像头凭据：保留硬编码默认，不外部化
CAMERA_IP = "192.168.2.222"
USERNAME = "root"
PASSWORD = "1234qwer"

CONF_THRESHOLD = _env("CONF_THRESHOLD", 0.3, float)
IOU_THRESHOLD = _env("IOU_THRESHOLD", 0.3, float)          # 连续帧匹配 IoU
NMS_IOU_THRESHOLD = _env("NMS_IOU_THRESHOLD", 0.5, float)  # 检测 NMS 阈值（CPU/NPU 统一）
DEBUG_MODE = _env("DEBUG_MODE", False, lambda v: str(v).lower() in ("1", "true", "yes", "on"))  # 生产默认 False：非鸟不入库
SAVE_COOLDOWN = _env("SAVE_COOLDOWN", 2, int)

DETECT_EVERY_N_FRAMES = _env("DETECT_EVERY_N_FRAMES", 2, int)
CONSECUTIVE_FRAMES_REQUIRED = _env("CONSECUTIVE_FRAMES_REQUIRED", 1, int)

# 检测后端：cpu（默认，兼容性强）| npu（Rockchip NPU / float16 RKNN，约 12x 更快、精度一致）
DETECT_BACKEND = _env("DETECT_BACKEND", "cpu", str).lower()
NPU_MODEL_PATH = _env("NPU_MODEL_PATH", "/home/birding/yolov8n_fp16.rknn", str)
CPU_MODEL_PATH = _env("CPU_MODEL_PATH", "/home/birding/yolov8n.pt", str)

# RTSP 拉流：TCP 传输抗丢包；RTSP_STREAM 可切子码流（如 ch0_1.h264）降低解码压力与延迟
RTSP_TRANSPORT = _env("RTSP_TRANSPORT", "tcp", str).lower()
RTSP_STREAM = _env("RTSP_STREAM", "ch0_0.h264", str)
RTSP_READ_TIMEOUT_SEC = _env("RTSP_READ_TIMEOUT_SEC", 10, int)
# rtsp_transport=tcp 抗丢包；stimeout(μs) 保证死流时 read 在超时后返回，拉流线程不会永久阻塞
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      f"rtsp_transport;{RTSP_TRANSPORT}|stimeout;{RTSP_READ_TIMEOUT_SEC * 1000000}")

# 运行时参数持久化（重启/断电后恢复，避免静默回退默认值）
CONFIG_PATH = Path(_env("CONFIG_PATH", str(Path(__file__).resolve().parent / "config.json")))

WORK_START = dtime(_env("WORK_START_HOUR", 5, int), 0)
WORK_END = dtime(_env("WORK_END_HOUR", 17, int), 0)

IMAGE_ROOT = Path(_env("IMAGE_ROOT", "/home/bird_image"))
IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

# ======================
# 运行时参数持久化
# ======================
def _save_config():
    try:
        data = {
            "camera_ip": CAMERA_IP,
            "conf": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "debug_mode": DEBUG_MODE,
            "save_cooldown": SAVE_COOLDOWN,
            "detect_every_n_frames": DETECT_EVERY_N_FRAMES,
            "consecutive_frames_required": CONSECUTIVE_FRAMES_REQUIRED,
            "detect_backend": DETECT_BACKEND,
            "work_start_hour": WORK_START.hour,
            "work_end_hour": WORK_END.hour,
        }
        tmp = CONFIG_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(CONFIG_PATH))
    except Exception as e:
        print(f"[{datetime.now()}] [警告] 参数持久化失败: {e}")

def _load_config():
    """启动时加载持久化参数（仅覆盖存在的键；凭据不持久化）。"""
    global CAMERA_IP, CONF_THRESHOLD, IOU_THRESHOLD, DEBUG_MODE, SAVE_COOLDOWN
    global DETECT_EVERY_N_FRAMES, CONSECUTIVE_FRAMES_REQUIRED, DETECT_BACKEND
    global WORK_START, WORK_END
    try:
        if not CONFIG_PATH.exists():
            return
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        if isinstance(data.get("camera_ip"), str) and re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", data["camera_ip"]):
            CAMERA_IP = data["camera_ip"]
        v = data.get("conf");
        if isinstance(v, (int, float)): CONF_THRESHOLD = max(0.1, min(0.9, float(v)))
        v = data.get("iou_threshold")
        if isinstance(v, (int, float)): IOU_THRESHOLD = max(0.1, min(0.9, float(v)))
        if isinstance(data.get("debug_mode"), bool): DEBUG_MODE = data["debug_mode"]
        v = data.get("save_cooldown")
        if isinstance(v, int) and v > 0: SAVE_COOLDOWN = v
        v = data.get("detect_every_n_frames")
        if isinstance(v, int) and 1 <= v <= 10: DETECT_EVERY_N_FRAMES = v
        v = data.get("consecutive_frames_required")
        if isinstance(v, int) and 1 <= v <= 5: CONSECUTIVE_FRAMES_REQUIRED = v
        if data.get("detect_backend") in ("cpu", "npu"): DETECT_BACKEND = data["detect_backend"]
        vs, ve = data.get("work_start_hour"), data.get("work_end_hour")
        if isinstance(vs, int) and 0 <= vs <= 23 and isinstance(ve, int) and 0 <= ve <= 23 and vs < ve:
            WORK_START = dtime(vs, 0); WORK_END = dtime(ve, 0)
        print(f"[{datetime.now()}] [配置] 已加载持久化参数: {CONFIG_PATH}")
    except Exception as e:
        print(f"[{datetime.now()}] [警告] 读取持久化参数失败: {e}")

_load_config()

# ======================
# 全局状态
# ======================
latest_frame = None
latest_frame_ts = 0
latest_boxes = []
latest_confidences = []

last_save_time = {}

capture_thread = None
stop_event = threading.Event()
restart_event = threading.Event()  # 用于触发摄像头重新连接

system_state = "INIT"
frame_counter = 0
save_counter = 0

state_lock = threading.Lock()  # 保护跨线程共享状态（采集线程 <-> 视频流生成器）

# ======================
# 检测后端（可插拔：CPU / NPU）
# ======================
BIRD_CLASS_ID = 14

class Box:
    __slots__ = ("cls", "conf", "xyxy")
    def __init__(self, cls, conf, xyxy):
        self.cls = cls          # COCO 类别 id
        self.conf = conf        # 置信度 0~1
        self.xyxy = xyxy        # 原始画面坐标 (x1,y1,x2,y2)

def _iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1]); xb = min(a[2], b[2]); yb = min(a[3], b[3])
    iw = max(0.0, xb - xa); ih = max(0.0, yb - ya)
    inter = iw * ih
    aa = max(1e-6, (a[2]-a[0])*(a[3]-a[1])); ab = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
    return inter / (aa + ab - inter)

def _nms(boxes, scores, thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        rest = order[1:]
        if len(rest) == 0: break
        order = rest[np.array([_iou(boxes[i], boxes[j]) for j in rest]) < thr]
    return keep

class CPUDetector:
    """CPU 后端：复用 ultralytics YOLO（默认，兼容性最好）。"""
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"[{datetime.now()}] [模型] CPU后端加载成功: {model_path}")
    def detect(self, frame):
        result = self.model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, iou=NMS_IOU_THRESHOLD, verbose=False, device='cpu')[0]
        out = []
        if result.boxes:
            for b in result.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                out.append(Box(int(b.cls), float(b.conf), (x1, y1, x2, y2)))
        return out

class NPUDetector:
    """NPU 后端：Rockchip RKNNLite + float16 模型，精度与 CPU 一致，速度约 12x。"""
    def __init__(self, model_path):
        # rknnlite 必须在 torch/ultralytics 之后导入，否则会破坏 logging 模块
        from ultralytics import YOLO  # noqa: F401  (仅用于保证导入顺序)
        from rknnlite.api import RKNNLite
        self.rk = RKNNLite()
        self.rk.load_rknn(model_path)
        self.rk.init_runtime()
        self.model_path = model_path
        # RGA 2D 加速（可选）：letterbox 预处理交给硬件，失败自动回退 cv2
        self.rga = None
        self.rga_ok = False
        if _env("RGA_ENABLED", 1, int) == 1:
            try:
                from rga_accel import RGAAccel, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888
                acc = RGAAccel()
                if acc.ok:
                    self.rga = acc
                    self.rga_ok = True
                    self.fmt_bgr = RK_FORMAT_BGR_888
                    self.fmt_rgb = RK_FORMAT_RGB_888
                    print(f"[{datetime.now()}] [模型] RGA 加速已启用 (rga_shim)" )
                else:
                    print(f"[{datetime.now()}] [模型] RGA 不可用，回退 cv2 预处理: {acc.err}")
            except Exception as e:
                print(f"[{datetime.now()}] [模型] RGA 初始化异常，回退 cv2 预处理: {e}")
        print(f"[{datetime.now()}] [模型] NPU后端加载成功: {model_path}")
    def _decode(self, box_raw, score_raw, conf, iou):
        # box_raw: (4,8400) xywh; score_raw: (80,8400) 已是概率(0~1)，无需再 sigmoid
        box = box_raw.T
        scores = score_raw.T
        xc, yc, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        bx = np.stack([xc - w/2, yc - h/2, xc + w/2, yc + h/2], 1)
        final = []
        for c in range(80):
            s = scores[:, c]
            idx = np.where(s > conf)[0]
            if len(idx) == 0: continue
            for k in _nms(bx[idx], s[idx], iou):
                final.append((c, float(s[idx][k]),
                              (float(bx[idx[k],0]), float(bx[idx[k],1]),
                               float(bx[idx[k],2]), float(bx[idx[k],3]))))
        return final
    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(640 / w, 640 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x = (640 - nw) // 2; pad_y = (640 - nh) // 2
        if self.rga_ok:
            # RGA 硬件路径：BGR 整帧 → RGB 640 letterbox（填边+缩放两步都在 RGA 内）
            rc = self.rga.letterbox(frame, w, h, self.fmt_bgr,
                                    canvas, 640, 640, self.fmt_rgb,
                                    114, pad_x, pad_y, nw, nh)
            if rc != 0:
                print(f"[{datetime.now()}] [警告] RGA letterbox 失败 rc={rc}，本次起回退 cv2")
                self.rga_ok = False
        if not self.rga_ok:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas[pad_y:pad_y+nh, pad_x:pad_x+nw, :] = resized
        inp = np.expand_dims(canvas, 0)  # (1,640,640,3) uint8 NHWC
        out = self.rk.inference(inputs=[inp], data_format='nhwc')[0]
        out = np.squeeze(out)            # (84,8400)
        dets = self._decode(out[:4, :], out[4:, :], CONF_THRESHOLD, NMS_IOU_THRESHOLD)
        result = []
        for cls, conf, (x1, y1, x2, y2) in dets:
            ox1 = (x1 - pad_x) / scale; oy1 = (y1 - pad_y) / scale
            ox2 = (x2 - pad_x) / scale; oy2 = (y2 - pad_y) / scale
            result.append(Box(cls, conf, (
                max(0, int(ox1)), max(0, int(oy1)),
                min(w, int(ox2)), min(h, int(oy2)))))
        return result

detector = None
def get_detector():
    global detector, DETECT_BACKEND
    if detector is None:
        detector, actual = _build_detector()
        # 如实回写实际后端：NPU 加载失败回退 CPU 时，/status 不再虚报 npu
        DETECT_BACKEND = actual
    return detector

def _build_detector():
    """返回 (detector, actual_backend)。NPU 构建失败时安全回退 CPU，并如实返回实际后端。"""
    if DETECT_BACKEND == "npu":
        try:
            return NPUDetector(NPU_MODEL_PATH), "npu"
        except Exception as e:
            print(f"[{datetime.now()}] [警告] NPU后端加载失败，回退CPU: {e}")
            traceback.print_exc()
            return CPUDetector(CPU_MODEL_PATH), "cpu"
    return CPUDetector(CPU_MODEL_PATH), "cpu"

def reload_detector():
    global detector
    detector, actual = _build_detector()
    return actual

def rtsp_url():
    return f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/{RTSP_STREAM}"

def rtsp_url_masked():
    return f"rtsp://{USERNAME}:***@{CAMERA_IP}:554/{RTSP_STREAM}"

def in_work_time():
    now = datetime.now().time()
    return WORK_START <= now <= WORK_END

def get_position_hash(box):
    try:
        x1, y1, x2, y2 = box
        grid_x = (x1 + x2) // 2 // 100
        grid_y = (y1 + y2) // 2 // 100
        return f"{grid_x}_{grid_y}"
    except Exception:
        return "0_0"

def compute_iou(box1, box2):
    try:
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union = box1_area + box2_area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
    except Exception:
        return 0.0

def filter_new_detections(boxes):
    """按位置去重：仅丢弃仍处于冷却中的位置，新位置仍可保存（修复原逻辑整体漏存问题）。"""
    global last_save_time
    current_time = time.time()
    # 定期清理过期记录，防止长期运行下字典无限增长
    if len(last_save_time) > 256:
        last_save_time = {k: v for k, v in last_save_time.items()
                          if current_time - v < max(600, SAVE_COOLDOWN * 10)}
    new_boxes = []
    if not boxes:
        return new_boxes
    for box in boxes:
        pos_hash = get_position_hash(box)
        if pos_hash in last_save_time and (current_time - last_save_time[pos_hash]) < SAVE_COOLDOWN:
            if DEBUG_MODE:
                print(f"  [去重] 位置{pos_hash}冷却中")
            continue
        last_save_time[pos_hash] = current_time
        new_boxes.append(box)
    return new_boxes

class ConsecutiveDetector:
    def __init__(self, required_frames=1):
        self.required_frames = required_frames
        self.current_streak = 0
        self.last_boxes = []

    def update(self, boxes):
        if not boxes:
            self.current_streak = 0
            self.last_boxes = []
            return False, []

        if self.required_frames <= 1:
            # 首帧即确认；同位置冷却统一交给 filter_new_detections 的 SAVE_COOLDOWN 处理
            # （修复：原 confirmed_hashes 永不过期，导致同一位置整个时段只存 1 张）
            return True, boxes

        if self.last_boxes:
            overlap_count = 0
            for box in boxes:
                for last_box in self.last_boxes:
                    if compute_iou(box, last_box) > IOU_THRESHOLD:
                        overlap_count += 1
                        break
            if overlap_count >= len(boxes) * 0.3:
                self.current_streak += 1
            else:
                self.current_streak = 1
        else:
            self.current_streak = 1

        self.last_boxes = boxes.copy()

        if self.current_streak >= self.required_frames:
            return True, boxes
        return False, []

    def reset(self):
        self.current_streak = 0
        self.last_boxes = []

consecutive_detector = ConsecutiveDetector(required_frames=CONSECUTIVE_FRAMES_REQUIRED)

def save_frame(frame, boxes=None, confidences=None, prefix=""):
    global save_counter
    if frame is None:
        print(f"[{datetime.now()}] [错误] 帧为空")
        return None
    try:
        day = datetime.now().strftime("%Y%m%d")
        base = IMAGE_ROOT / day
        thumb = base / "thumb"
        base.mkdir(parents=True, exist_ok=True)
        thumb.mkdir(parents=True, exist_ok=True)
        img = frame.copy()
        if boxes:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                try:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    if confidences and i < len(confidences):
                        label = f"{confidences[i]:.2f}"
                        cv2.putText(img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"  [警告] 绘制框失败: {e}")
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        name = f"{prefix}{timestamp}.jpg"
        filepath = base / name
        success = cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            print(f"[{datetime.now()}] [错误] cv2.imwrite失败: {filepath}")
            return None
        try:
            h, w = img.shape[:2]
            scale = 240 / max(h, w)
            thumb_img = cv2.resize(img, (int(w*scale), int(h*scale)))
            cv2.imwrite(str(thumb / name), thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        except Exception as e:
            print(f"  [警告] 缩略图生成失败: {e}")
        save_counter += 1
        file_size = filepath.stat().st_size / 1024
        print(f"[{datetime.now()}] [保存成功] {name} ({file_size:.1f}KB), 总计:{save_counter}")
        return str(filepath)
    except Exception as e:
        print(f"[{datetime.now()}] [错误] 保存异常: {e}")
        traceback.print_exc()
        return None

class FrameReader(threading.Thread):
    """独立拉流线程：持续 cap.read() 只保留最新帧，推理不再阻塞读流，
    避免检测耗时导致 RTSP 缓冲积压、预览/截图延迟越拉越大（修复 P1-5）。"""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0
        self.started_ts = time.time()   # 首帧到达前 idle 以启动时间为基准，避免误判死流
        self.stopped = threading.Event()

    def run(self):
        fail = 0
        while not self.stopped.is_set():
            ret, frame = self.cap.read()
            if not ret:
                fail += 1
                if fail > 30:            # 连续读取失败视为流断开
                    return
                time.sleep(0.02)
                continue
            fail = 0
            with self.lock:
                self.frame = frame
                self.ts = time.time()

    def get(self, max_age=0.5):
        with self.lock:
            if self.frame is None or time.time() - self.ts > max_age:
                return None
            return self.frame.copy()   # 返回副本，推理/保存可安全持有

    def idle_seconds(self):
        ref = self.ts if self.ts > 0 else self.started_ts
        return time.time() - ref

    def stop(self):
        self.stopped.set()

def capture_loop():
    global latest_frame, latest_frame_ts, latest_boxes, latest_confidences
    global system_state, frame_counter, consecutive_detector, detector
    global CAMERA_IP, DETECT_BACKEND  # 确保使用最新值

    print(f"[{datetime.now()}] [启动] 观鸟系统启动")
    print(f"[{datetime.now()}] [配置] 阈值:{CONF_THRESHOLD}, 间隔:{DETECT_EVERY_N_FRAMES}帧, 确认:{CONSECUTIVE_FRAMES_REQUIRED}帧")
    print(f"[{datetime.now()}] [配置] 图片保存路径: {IMAGE_ROOT}")

    try:
        detector = get_detector()   # NPU 失败会回退 CPU，DETECT_BACKEND 回写为实际后端
    except Exception as e:
        print(f"[{datetime.now()}] [致命错误] 模型加载失败，服务无法启动: {e}")
        system_state = "ERROR"
        return

    retry_count = 0

    while not stop_event.is_set():
        # 检查是否需要重启（IP 切换触发）
        if restart_event.is_set():
            print(f"[{datetime.now()}] [重启] 检测到IP变更，重新连接摄像头...")
            restart_event.clear()
            system_state = "RESTARTING"
            time.sleep(1)

        if not in_work_time():
            if system_state != "SLEEPING":
                print(f"[{datetime.now()}] [状态] 进入休眠时段")
                system_state = "SLEEPING"
            stop_event.wait(30)
            continue

        system_state = "OFFLINE"
        current_ip = CAMERA_IP  # 记录当前IP，用于检测变化

        try:
            cap = cv2.VideoCapture(rtsp_url(), cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"[{datetime.now()}] [错误] VideoCapture创建失败: {e}")
            stop_event.wait(5)
            continue

        if not cap.isOpened():
            # 无限重试 + 指数退避（2s→60s 封顶）；修复原 100 次后线程永久死亡问题
            retry_count += 1
            delay = min(60, 2 ** min(retry_count, 6))
            print(f"[{datetime.now()}] [错误] 无法连接摄像头({retry_count})，{delay}秒后重试... [{rtsp_url_masked()}]")
            stop_event.wait(delay)
            continue

        print(f"[{datetime.now()}] [状态] 摄像头连接成功: {CAMERA_IP} ({RTSP_TRANSPORT.upper()} / {RTSP_STREAM})")
        system_state = "RUNNING"
        retry_count = 0
        frame_counter = 0
        consecutive_detector.reset()

        reader = FrameReader(cap)
        reader.start()

        while not stop_event.is_set() and in_work_time():
            # 检查IP是否被修改
            if CAMERA_IP != current_ip or restart_event.is_set():
                print(f"[{datetime.now()}] [切换] IP变更 {current_ip} -> {CAMERA_IP}")
                break  # 跳出内层循环，重新连接

            # 拉流线程死亡或长时间无新帧 → 重连
            if not reader.is_alive() or reader.idle_seconds() > 10:
                print(f"[{datetime.now()}] [错误] 拉流线程异常/无新帧，重新连接...")
                break

            try:
                frame = reader.get(max_age=2.0)
                if frame is None:
                    time.sleep(0.02)
                    continue

                frame_counter += 1
                with state_lock:
                    latest_frame = frame
                    latest_frame_ts = time.time()

                if frame_counter % DETECT_EVERY_N_FRAMES != 0:
                    continue

                boxes = []
                confidences = []

                try:
                    detections = detector.detect(frame)
                    for b in detections:
                        try:
                            cls = b.cls
                            conf = b.conf
                            if cls != BIRD_CLASS_ID and not DEBUG_MODE:
                                continue
                            x1, y1, x2, y2 = b.xyxy
                            if (x2 - x1) < 20 or (y2 - y1) < 20:
                                continue
                            boxes.append((x1, y1, x2, y2))
                            confidences.append(conf)
                        except Exception:
                            continue

                    if DEBUG_MODE and boxes:
                        print(f"[{datetime.now()}] [检测] 发现{len(boxes)}个目标, 置信度:{[f'{c:.2f}' for c in confidences]}")

                except Exception as e:
                    print(f"[{datetime.now()}] [错误] 推理异常: {e}")
                    continue

                with state_lock:
                    latest_boxes = boxes
                    latest_confidences = confidences

                if boxes:
                    try:
                        should_save, boxes_to_save = consecutive_detector.update(boxes)
                        if should_save:
                            new_boxes = filter_new_detections(boxes_to_save)
                            if new_boxes:
                                conf_by_box = {id(b): c for b, c in zip(boxes, confidences)}
                                new_confs = [conf_by_box.get(id(b), 0.0) for b in new_boxes]
                                filepath = save_frame(frame, new_boxes, new_confs, prefix="auto_")
                                if not filepath:
                                    print(f"[{datetime.now()}] [错误] 保存失败")
                            elif DEBUG_MODE:
                                print(f"  [跳过] 去重过滤")
                        else:
                            if DEBUG_MODE:
                                print(f"  [等待] 连续帧确认中...")
                    except Exception as e:
                        print(f"[{datetime.now()}] [错误] 保存逻辑异常: {e}")
                        traceback.print_exc()

            except Exception as e:
                print(f"[{datetime.now()}] [错误] 主循环异常: {e}")
                traceback.print_exc()
                time.sleep(1)

        # 停止拉流线程并释放摄像头（FFMPEG stimeout 保证阻塞的 read 会在超时后返回）
        try:
            reader.stop()
            reader.join(timeout=12)
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass

        if system_state != "RESTARTING":
            print(f"[{datetime.now()}] [状态] 摄像头断开")

        system_state = "OFFLINE"
        consecutive_detector.reset()
        stop_event.wait(2)

@asynccontextmanager
async def lifespan(_app):
    # 停机时置位 stop_event：MJPEG 生成器与采集线程随之退出，uvicorn 可秒级优雅退出
    # （修复：原模块级 signal 处理器被 uvicorn 覆盖，stop_event 永不置位，重启被 systemd 强杀）
    yield
    stop_event.set()
    print(f"[{datetime.now()}] [停机] lifespan shutdown，已通知采集线程与视频流退出")

app = FastAPI(title="Birding Camera System v5", lifespan=lifespan)
# 内网服务：关闭 credentials，避免与 allow_origins=['*'] 冲突且无实际用途
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory=str(IMAGE_ROOT)), name="images")
app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="frontend")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "conf": CONF_THRESHOLD,
        "camera_ip": CAMERA_IP,
        "state": system_state,
        "start": WORK_START.hour,
        "end": WORK_END.hour,
        "detect_interval": DETECT_EVERY_N_FRAMES,
        "consecutive_required": CONSECUTIVE_FRAMES_REQUIRED
    })

@app.get("/healthz")
def healthz():
    unhealthy = system_state in ("ERROR", "DEAD")
    return JSONResponse(
        content={"status": "unhealthy" if unhealthy else "ok", "state": system_state},
        status_code=503 if unhealthy else 200,
    )

@app.get("/status")
def status():
    return {
        "state": system_state,
        "backend": DETECT_BACKEND,
        "conf": CONF_THRESHOLD,
        "camera": CAMERA_IP,
        "work_start": f"{WORK_START.hour:02d}:00",
        "work_end": f"{WORK_END.hour:02d}:00",
        "detect_interval": DETECT_EVERY_N_FRAMES,
        "consecutive_required": CONSECUTIVE_FRAMES_REQUIRED,
        "frame_counter": frame_counter,
        "save_counter": save_counter,
        "latest_boxes": len(latest_boxes)
    }

@app.post("/update_camera")
def update_camera(ip: str = Form(...)):
    global CAMERA_IP
    ip = ip.strip()
    if not re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", ip) or any(int(o) > 255 for o in ip.split(".")):
        return {"ok": False, "message": "IP 格式无效，应为如 192.168.2.222 的 IPv4 地址"}
    old_ip = CAMERA_IP
    CAMERA_IP = ip
    print(f"[{datetime.now()}] [配置] 摄像头IP: {old_ip} -> {ip}")

    # 触发重新连接
    restart_event.set()
    _save_config()

    return {"ok": True, "ip": CAMERA_IP, "message": f"已切换到 {ip}，正在重新连接..."}

@app.post("/update_yolo")
def update_yolo(conf: float = Form(...)):
    global CONF_THRESHOLD
    CONF_THRESHOLD = max(0.1, min(0.9, conf))
    print(f"[{datetime.now()}] [配置] 阈值改为: {CONF_THRESHOLD}")
    _save_config()
    return {"ok": True, "message": f"YOLO 置信度已更新为 {CONF_THRESHOLD:.2f}"}

@app.post("/update_backend")
def update_backend(backend: str = Form(...)):
    """运行时切换检测后端：cpu | npu（无需重启进程）。若请求 npu 但不可用，如实回退并提示。"""
    global DETECT_BACKEND
    backend = backend.strip().lower()
    if backend not in ("cpu", "npu"):
        return {"ok": False, "message": "后端仅支持 cpu 或 npu"}
    prev = DETECT_BACKEND
    DETECT_BACKEND = backend
    try:
        actual = reload_detector()
    except Exception as e:
        DETECT_BACKEND = prev
        print(f"[{datetime.now()}] [错误] 后端切换失败: {e}")
        traceback.print_exc()
        return {"ok": False, "message": f"后端切换失败: {e}"}
    if actual != backend:
        # 请求的 NPU 不可用，已静默回退 CPU —— 如实回报，避免 UI 误显示 NPU
        DETECT_BACKEND = actual
        _save_config()
        return {"ok": True, "backend": actual, "fallback": True,
                "message": f"{backend.upper()} 不可用，已回退 {actual.upper()}"}
    _save_config()
    return {"ok": True, "backend": actual, "message": f"检测后端已切换为 {actual.upper()}"}

# 不同后端的优化推荐参数（基于实测性能：CPU ~0.1 FPS，NPU ~1.7 FPS / ~12x）
RECOMMENDED_PARAMS = {
    "cpu": {
        "conf": 0.30, "interval": 4, "consecutive": 2,
        "note": "CPU 推理慢（约 0.1 FPS），降低检测频率以省算力；停留型鸟类仍可被多帧覆盖。",
    },
    "npu": {
        "conf": 0.35, "interval": 1, "consecutive": 2,
        "note": "NPU 推理快（约 1.7 FPS，约 12×），每帧检测以最快响应；精度足够可略提阈值压低误报。",
    },
}

@app.get("/recommended_params")
def recommended_params():
    return RECOMMENDED_PARAMS

@app.post("/update_worktime")
def update_worktime(start: int = Form(...), end: int = Form(...)):
    global WORK_START, WORK_END
    if not (0 <= start <= 23 and 0 <= end <= 23):
        return {"ok": False, "message": "小时数应在 0-23 之间"}
    if start >= end:
        return {"ok": False, "message": "开始小时必须早于结束小时（暂不支持跨夜时段）"}
    WORK_START = dtime(start, 0)
    WORK_END = dtime(end, 0)
    _save_config()
    return {"ok": True, "message": f"观察时段已更新: {start:02d}:00 - {end:02d}:00"}

@app.post("/update_detection_params")
def update_detection_params(interval: int = Form(...), consecutive: int = Form(...)):
    global DETECT_EVERY_N_FRAMES, CONSECUTIVE_FRAMES_REQUIRED, consecutive_detector
    DETECT_EVERY_N_FRAMES = max(1, min(10, interval))
    CONSECUTIVE_FRAMES_REQUIRED = max(1, min(5, consecutive))
    consecutive_detector = ConsecutiveDetector(required_frames=CONSECUTIVE_FRAMES_REQUIRED)
    print(f"[{datetime.now()}] [配置] 检测间隔:{DETECT_EVERY_N_FRAMES}, 确认帧数:{CONSECUTIVE_FRAMES_REQUIRED}")
    _save_config()
    return {"ok": True, "message": f"检测参数已更新: 每{DETECT_EVERY_N_FRAMES}帧检测，需连续{CONSECUTIVE_FRAMES_REQUIRED}帧确认"}

@app.post("/snapshot")
def snapshot():
    if latest_frame is not None and latest_boxes:
        filepath = save_frame(latest_frame, latest_boxes, latest_confidences, prefix="snap_")
        if filepath:
            return {"ok": True, "path": filepath}
    return {"ok": False, "reason": "最近未检测到鸟类"}

@app.post("/snapshot_simple")
def snapshot_simple():
    if latest_frame is not None:
        filepath = save_frame(latest_frame, boxes=None, prefix="manual_")
        if filepath:
            return {"ok": True, "path": filepath}
    return {"ok": False, "reason": "无法获取当前画面"}

@app.get("/api/photos")
def get_photos(date: str = Query(None), limit: int = Query(50)):
    # date 仅接受 YYYYMMDD，防路径穿越；limit 加上限
    if not (date and re.match(r"^\d{8}$", date)):
        return JSONResponse(status_code=400, content={"error": "invalid date, expected YYYYMMDD"})
    limit = max(1, min(limit, 200))
    base_path = IMAGE_ROOT / date
    if not base_path.exists():
        return {"date": date, "photos": [], "count": 0}
    photos = []
    try:
        for img_file in sorted(base_path.glob("*.jpg")):
            if img_file.name == "thumb":
                continue
            try:
                stat = img_file.stat()
                name_parts = img_file.stem.split('_')
                time_str = name_parts[1] if len(name_parts) > 1 else img_file.stem[:6]
                photos.append({"filename": img_file.name, "time": time_str, "thumb_url": f"/images/{date}/thumb/{img_file.name}", "full_url": f"/images/{date}/{img_file.name}", "size": stat.st_size, "timestamp": stat.st_mtime})
            except Exception as e:
                continue
        photos.sort(key=lambda x: x["timestamp"], reverse=True)
        photos = photos[:limit]
    except Exception as e:
        print(f"[错误] 读取照片列表失败: {e}")
    return {"date": date, "count": len(photos), "photos": photos}

@app.get("/api/dates")
def get_dates():
    dates = []
    try:
        for item in sorted(IMAGE_ROOT.iterdir()):
            if item.is_dir() and item.name.isdigit() and len(item.name) == 8:
                try:
                    count = len(list(item.glob("*.jpg")))
                    if count > 0:
                        dates.append({"date": item.name, "display": f"{item.name[:4]}-{item.name[4:6]}-{item.name[6:]}", "count": count})
                except Exception:
                    continue
        dates.reverse()
    except Exception as e:
        print(f"[错误] 读取日期列表失败: {e}")
    return {"dates": dates}

@app.get("/video_feed")
def video_feed():
    def gen():
        while not stop_event.is_set():
            try:
                with state_lock:
                    if latest_frame is None or time.time() - latest_frame_ts > 2:
                        time.sleep(0.1)
                        continue
                    frame = latest_frame.copy()
                    boxes = list(latest_boxes)
                    confs = list(latest_confidences)
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    try:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        if i < len(confs):
                            label = f"{confs[i]:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception:
                        continue
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                time.sleep(0.1)   # 预览限帧 10fps：编码是观看端 CPU 大头，预览无需 20fps
            except Exception as e:
                time.sleep(0.1)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

# ======================
# 启动采集线程 & 入口
# ======================
# 停机处理由 lifespan 完成：uvicorn 会接管 SIGTERM/SIGINT，模块级 signal 处理器
# 在两种启动方式下都不会被触发（死代码），已移除。

capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

if __name__ == "__main__":
    import uvicorn
    # access_log=False：前端每秒轮询 /status，避免 journal 每天堆数万行访问日志
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
