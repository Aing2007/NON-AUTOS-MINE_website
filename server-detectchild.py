# server.py
import os, io, time, asyncio, json, struct
import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

# ---------------------------
# Performance tweaks (CPU)
# ---------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
cv2.setNumThreads(0)

app = FastAPI()

# ---------------------------
# Load model (once)
# ---------------------------
MODEL_PATH = os.getenv("ANIMAL_MODEL", "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/hand_detection.pt")

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(MODEL_PATH)
model.to(device)
# FP16 บน CUDA จะไวขึ้น
use_half = (device == "cuda")
if use_half:
    model.half()

# warmup
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
_ = model.predict(dummy, imgsz=640, conf=0.25, iou=0.45, device=device, verbose=False)

# class names
NAMES = model.names

# ---------------------------
# Utils
# ---------------------------
def jpeg_to_ndarray(jpeg_bytes: bytes) -> np.ndarray:
    npbuf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    return img

def to_boxes_json(result, img_w: int, img_h: int):
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy.detach().cpu().numpy()  # [N,4]
    conf = result.boxes.conf.detach().cpu().numpy()  # [N]
    cls  = result.boxes.cls.detach().cpu().numpy()   # [N]
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        c = float(conf[i])
        k = int(cls[i])
        label = NAMES.get(k, str(k)) if isinstance(NAMES, dict) else (NAMES[k] if k < len(NAMES) else str(k))
        # limit to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        w = max(0, min(w, img_w - x1))
        h = max(0, min(h, img_h - y1))
        boxes.append({
            "x": float(x1), "y": float(y1), "w": float(w), "h": float(h),
            "cls": k, "conf": c, "label": label
        })
    return boxes

# ---------------------------
# WebSocket endpoint
# Packet format from client:
# [hdrLen(uint32 LE) | hdr(JSON bytes) | JPEG bytes]
# hdr: {"t0": <client send time ms>}
# ---------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    latest_frame = None
    latest_ts = 0.0
    queue_event = asyncio.Event()
    sending_lock = asyncio.Lock()

    async def receiver():
        nonlocal latest_frame, latest_ts
        try:
            while True:
                data = await ws.receive_bytes()
                # parse header length (4 bytes little-endian)
                if len(data) < 4:
                    continue
                hdr_len = struct.unpack("<I", data[:4])[0]
                if 4 + hdr_len > len(data):
                    continue
                hdr_bytes = data[4:4+hdr_len]
                jpg_bytes = data[4+hdr_len:]

                try:
                    hdr = json.loads(hdr_bytes.decode("utf-8"))
                    t0 = float(hdr.get("t0", 0.0))
                except Exception:
                    t0 = 0.0

                # keep-latest (drop old)
                latest_frame = (jpg_bytes, t0)
                latest_ts = time.perf_counter()
                queue_event.set()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            queue_event.set()  # ปลุก consumer ให้จบ

    async def consumer():
        # วนอ่าน latest_frame เมื่อมีงานใหม่เท่านั้น
        while True:
            await queue_event.wait()
            queue_event.clear()
            if latest_frame is None:
                break
            jpg_bytes, t0 = latest_frame
            # decode
            img = jpeg_to_ndarray(jpg_bytes)
            if img is None:
                continue

            h, w = img.shape[:2]
            t_start = time.perf_counter()
            # inference
            with torch.inference_mode():
                # imgsz 640 ดีสุดสำหรับความเร็ว/ความแม่นยำทั่วไป
                res = model.predict(img, imgsz=640, conf=0.25, iou=0.45, device=device, half=use_half, verbose=False)[0]
            t_end = time.perf_counter()
            t_ms = (t_end - t_start) * 1000.0

            dets = to_boxes_json(res, w, h)
            # latency จาก client t0 → ส่งกลับ (คร่าวๆ)
            client_latency = 0.0
            if t0:
                client_latency = (time.time()*1000.0) - t0

            payload = {
                "boxes": dets,
                "w": w, "h": h,
                "t_ms": t_ms,
                "client_latency_ms": client_latency
            }
            msg = json.dumps(payload)
            # ส่งกลับ (กันส่งชนกัน)
            async with sending_lock:
                try:
                    await ws.send_text(msg)
                except Exception:
                    break

    recv_task = asyncio.create_task(receiver())
    cons_task = asyncio.create_task(consumer())

    await asyncio.gather(recv_task, cons_task, return_exceptions=True)



#source .venv/bin/activate
#pip install --upgrade pip
#pip install "fastapi[all]" uvicorn ultralytics opencv-python-headless torch torchvision

#cd Backend
# uvicorn server-detectchild:app --host 0.0.0.0 --port 8001
