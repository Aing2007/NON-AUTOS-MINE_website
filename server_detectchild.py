# Backend/server_detectchild.py
import os, json, struct, asyncio, time, datetime
import numpy as np, cv2, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

# ================= Performance hints =================
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
cv2.setNumThreads(0)

# ตั้ง threshold มากกว่า 50% (ปรับได้ทาง ENV ถ้าต้องการ)
MIN_SEND_CONF = float(os.getenv("MIN_SEND_CONF", "0.6"))  # เงื่อนไข: > 0.5

# ปรับ path โมเดลให้ตรงโปรเจกต์ของคุณ
MODEL_PATH = os.getenv("ANIMAL_MODEL", "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/animal_detection.pt")

# เลือกอุปกรณ์
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

app = FastAPI(title="Vision WS")

# โหลดโมเดลครั้งเดียว
model = YOLO(MODEL_PATH).to(device)
use_half = (device == "cuda")
if use_half:
    model.half()

# วอร์มอัพเบา ๆ
_ = model.predict(np.zeros((640,640,3), np.uint8), imgsz=640, device=device, verbose=False)
NAMES = model.names

# ================= Helpers =================
def jpg_to_nd(jpeg: bytes):
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def to_boxes(res, w, h):
    """
    แปลงผล YOLO เป็นรายการกล่องในพิกัดภาพ: {x,y,w,h,cls,conf,label}
    """
    out=[]
    if not getattr(res,"boxes",None) or len(res.boxes)==0:
        return out
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()
    cls  = res.boxes.cls.detach().cpu().numpy()
    for i in range(xyxy.shape[0]):
        x1,y1,x2,y2 = xyxy[i]
        ww = max(0, x2-x1); hh = max(0, y2-y1)
        k  = int(cls[i]); c = float(conf[i])
        label = NAMES.get(k,str(k)) if isinstance(NAMES,dict) else (NAMES[k] if k<len(NAMES) else str(k))
        # clip ให้อยู่ในภาพ
        x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
        ww = max(0, min(ww, w-x1)); hh = max(0, min(hh, h-y1))
        out.append({
            "x": float(x1), "y": float(y1), "w": float(ww), "h": float(hh),
            "cls": k, "conf": c, "label": label
        })
    return out

def log_detections(filtered_boxes, t_infer_ms):
    """
    พิมพ์ผลลง terminal เมื่อมีกรอบผ่านเกณฑ์
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    parts = [
        f"{b['label']} {b['conf']*100:.1f}%@({int(b['x'])},{int(b['y'])},{int(b['w'])},{int(b['h'])})"
        for b in filtered_boxes
    ]
    print(f"[{ts}] DETECT {len(filtered_boxes)} | " + " , ".join(parts) + f" | {t_infer_ms:.1f} ms", flush=True)

# ================= WebSocket (รองรับทั้ง /ws และ /vision/ws) =================
async def _vision_ws_impl(ws: WebSocket):
    await ws.accept()
    latest = None
    ev = asyncio.Event()
    lock = asyncio.Lock()

    # debug path & origin (ช่วยไล่ 403 ได้)
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

            t1 = time.perf_counter()
            with torch.inference_mode():
                res = model.predict(
                    img, imgsz=640, conf=0.25, iou=0.45,  # คงค่าพื้นฐานเดิม
                    device=device, half=use_half, verbose=False
                )[0]
            t2 = time.perf_counter()
            t_ms = (t2 - t1) * 1000.0

            all_boxes = to_boxes(res, w, h)
            filtered = [b for b in all_boxes if b["conf"] > MIN_SEND_CONF]  # > 50%

            if not filtered:
                # ส่ง heartbeat ล้าง overlay (ไม่ส่ง "กรอบ" ใด ๆ เพราะต่ำกว่า threshold)
                payload = {"boxes": [], "w": w, "h": h, "t_ms": t_ms}
                if t0:
                    payload["client_latency_ms"] = (time.time()*1000.0) - t0
                msg = json.dumps(payload)
                async with lock:
                    try:
                        await ws.send_text(msg)
                    except:
                        break
                continue

            # แสดงผลใน terminal เมื่อมีกรอบผ่านเกณฑ์
            try:
                log_detections(filtered, t_ms)
            except Exception:
                pass

            payload = {"boxes": filtered, "w": w, "h": h, "t_ms": t_ms}
            if t0:
                payload["client_latency_ms"] = (time.time()*1000.0) - t0

            msg = json.dumps(payload)
            async with lock:
                try:
                    await ws.send_text(msg)
                except:
                    break

    await asyncio.gather(asyncio.create_task(rx()), asyncio.create_task(tx()))

# ALIASES: รับได้ทั้ง /ws และ /vision/ws (ไม่ต้องแก้ frontend)
@app.websocket("/ws")
async def ws_short(ws: WebSocket):
    await _vision_ws_impl(ws)

@app.websocket("/vision/ws")
async def ws_long(ws: WebSocket):
    await _vision_ws_impl(ws)

# Health check (เดิม)
@app.get("/vision/health")
def health():
    return {"ok": True, "min_send_conf": MIN_SEND_CONF}


""""cd "NON-AUTOS-MINE(Overall)/Backend"
uvicorn server_detectchild:app --host 127.0.0.1 --port 8001 --reload"""