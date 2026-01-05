# server_dual_detect.py
# -*- coding: utf-8 -*-


import os, json, struct, asyncio, time, datetime
import numpy as np, cv2, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

# ================= Performance hints (เหมือนแม่แบบ) =================
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
cv2.setNumThreads(0)

# Threshold ส่งออก (กรองอีกชั้นหลัง YOLO.predict)
MIN_SEND_CONF = float(os.getenv("MIN_SEND_CONF", "0.6"))

# Path โมเดล (ตั้งผ่าน ENV)
HUMAN_MODEL_PATH  = os.getenv("HUMAN_MODEL",  "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/human_detection.pt")
ANIMAL_MODEL_PATH = os.getenv("ANIMAL_MODEL", "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/animal_detection.pt")

# อุปกรณ์
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
use_half = (device == "cuda")

app = FastAPI(title="Vision WS (Dual Models: human + animal)")

# ================ โหลดโมเดล (ครั้งเดียว) ================
model_human  = YOLO(HUMAN_MODEL_PATH).to(device)
model_animal = YOLO(ANIMAL_MODEL_PATH).to(device)
if use_half:
    model_human.half()
    model_animal.half()

# วอร์มอัพเบา ๆ เพื่อลดเฟรมแรกช้า
_ = model_human.predict(np.zeros((640,640,3), np.uint8), imgsz=640, device=device, verbose=False)
_ = model_animal.predict(np.zeros((640,640,3), np.uint8), imgsz=640, device=device, verbose=False)

NAMES_H = model_human.names
NAMES_A = model_animal.names

# ================= Helpers =================
def jpg_to_nd(jpeg: bytes):
    """แปลง JPEG bytes -> ndarray BGR"""
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def label_from_names(names, k: int) -> str:
    if isinstance(names, dict):
        return names.get(k, str(k))
    try:
        return names[k]
    except Exception:
        return str(k)

def to_boxes(res, w, h, names, src_tag: str):
    """
    แปลงผล YOLO เป็นรายการกล่อง: {x,y,w,h,cls,conf,label,src}
    พิกัดอยู่ในสเกลภาพอินพุต (เว็บจะ mirror เอง)
    """
    out=[]
    if not getattr(res, "boxes", None) or len(res.boxes) == 0:
        return out
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()
    cls  = res.boxes.cls.detach().cpu().numpy()
    for i in range(xyxy.shape[0]):
        x1,y1,x2,y2 = xyxy[i]
        ww = max(0, x2-x1); hh = max(0, y2-y1)
        k  = int(cls[i]); c = float(conf[i])
        label = label_from_names(names, k)
        # clip ให้อยู่ในภาพ
        x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
        ww = max(0, min(ww, w-x1)); hh = max(0, min(hh, h-y1))
        out.append({
            "x": float(x1), "y": float(y1), "w": float(ww), "h": float(hh),
            "cls": k, "conf": c, "label": label, "src": src_tag
        })
    return out

def log_detections(tag, boxes, t_ms):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    parts = [
        f"{b['src']}::{b['label']} {b['conf']*100:.1f}%@({int(b['x'])},{int(b['y'])},{int(b['w'])},{int(b['h'])})"
        for b in boxes
    ]
    print(f"[{ts}] {tag} {len(boxes)} | " + " , ".join(parts) + f" | {t_ms:.1f} ms", flush=True)

def _predict_once(model, img):
    """รัน YOLO.predict 1 ครั้ง และคืน (result, t_ms)"""
    t1 = time.perf_counter()
    with torch.inference_mode():
        res = model.predict(
            img, imgsz=640, conf=0.25, iou=0.45,
            device=device, half=use_half, verbose=False
        )[0]
    t_ms = (time.perf_counter() - t1) * 1000.0
    return res, t_ms

# ================= WebSocket (ตามแม่แบบ: /ws และ /vision/ws) =================
async def _vision_ws_impl(ws: WebSocket):
    await ws.accept()
    latest = None
    ev = asyncio.Event()
    send_lock = asyncio.Lock()

    # debug path & origin
    try:
        print(f"[WS] path={ws.scope.get('path')} origin={ws.headers.get('origin')}", flush=True)
    except Exception:
        pass

    async def rx():
        nonlocal latest
        try:
            while True:
                data = await ws.receive_bytes()
                if len(data) < 4:
                    continue
                hdr_len = struct.unpack("<I", data[:4])[0]
                if 4 + hdr_len > len(data):
                    continue
                t0 = 0.0
                try:
                    t0 = float(json.loads(data[4:4+hdr_len].decode()).get("t0", 0.0))
                except:
                    pass
                latest = (data[4+hdr_len:], t0)
                ev.set()
        except WebSocketDisconnect:
            pass
        finally:
            ev.set()

    async def tx():
        while True:
            await ev.wait()
            ev.clear()
            if latest is None:
                break

            jpg, t0 = latest
            img = jpg_to_nd(jpg)
            if img is None:
                continue
            h, w = img.shape[:2]

            # ===== รัน 2 โมเดล "พร้อมกัน" ด้วย asyncio.to_thread =====
            t_total_start = time.perf_counter()
            task_h = asyncio.to_thread(_predict_once, model_human, img)
            task_a = asyncio.to_thread(_predict_once, model_animal, img)
            res_h, t_h = await task_h
            res_a, t_a = await task_a
            t_total = (time.perf_counter() - t_total_start) * 1000.0

            # แปลงผลเป็นกล่อง
            boxes_h = to_boxes(res_h, w, h, NAMES_H, src_tag="human")
            boxes_a = to_boxes(res_a, w, h, NAMES_A, src_tag="animal")

            # กรองความมั่นใจ
            boxes_h = [b for b in boxes_h if b["conf"] > MIN_SEND_CONF]
            boxes_a = [b for b in boxes_a if b["conf"] > MIN_SEND_CONF]

            merged = boxes_h + boxes_a

            # log เฉพาะเมื่อมีผล
            if merged:
                try:
                    log_detections("DETECT", merged, t_total)
                except Exception:
                    pass

            # สร้าง payload (คง t_ms เดิมเพื่อความเข้ากันได้; ใส่เวลารายโมเดลเพิ่ม)
            payload = {
                "boxes": merged,
                "w": w, "h": h,
                "t_ms": t_total,            # backward-compatible
                "t_ms_total": t_total,
                "t_ms_human": t_h,
                "t_ms_animal": t_a,
                # เพิ่มข้อมูลเสริมแยก (หน้าเว็บเดิมไม่ใช้ แต่มีไว้ดีบัก)
                "boxes_human": boxes_h,
                "boxes_animal": boxes_a
            }
            if t0:
                payload["client_latency_ms"] = (time.time()*1000.0) - t0

            msg = json.dumps(payload)
            async with send_lock:
                try:
                    await ws.send_text(msg)
                except:
                    break

    await asyncio.gather(asyncio.create_task(rx()), asyncio.create_task(tx()))

# ALIASES: /ws และ /vision/ws (เหมือนแม่แบบ)
@app.websocket("/ws")
async def ws_short(ws: WebSocket):
    await _vision_ws_impl(ws)

@app.websocket("/vision/ws")
async def ws_long(ws: WebSocket):
    await _vision_ws_impl(ws)

# Health check (เพิ่มชื่อโมเดล)
@app.get("/vision/health")
def health():
    return {
        "ok": True,
        "device": device,
        "min_send_conf": MIN_SEND_CONF,
        "human_model": os.path.basename(HUMAN_MODEL_PATH),
        "animal_model": os.path.basename(ANIMAL_MODEL_PATH)
    }

"""
Backend WS (อิงโครงสร้างแม่แบบ) แต่รัน 2 โมเดลพร้อมกัน: HUMAN + ANIMAL
- โปรโตคอลจากเว็บ: binary packet = [uint32 header_len][header JSON UTF-8][JPEG bytes]
  header JSON อย่างน้อยมี {"t0": <client_timestamp_ms>}
- ตอบกลับเป็น text JSON:
  {
    "boxes": [ {x,y,w,h,label,conf,src}, ... ],  # รวมผลทั้งสองโมเดล (หน้าเว็บเดิมใช้ได้ทันที)
    "w": <width>, "h": <height>,
    "t_ms": <รวมเวลา>, "t_ms_total": <รวมเวลา>, "t_ms_human": <ms>, "t_ms_animal": <ms>,
    "client_latency_ms": <optional>
    // เสริม (เผื่อดีบัก): "boxes_human": [...], "boxes_animal": [...]
  }

วิธีรัน:
source .venv/bin/activate
cd Backend
uvicorn server_playgame:app --host 127.0.0.1 --port 8002 --reload"""