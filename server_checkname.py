import os, cv2, json, time, base64, asyncio, concurrent.futures
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ====== CONFIG ======
CONF_THRES = float(os.getenv("CONF", "0.35"))
IOU_THRES  = float(os.getenv("IOU",  "0.45"))
IMG_SIZE   = int(os.getenv("IMG",    "640"))

MODEL_PATH_ANIMAL = os.getenv("YOLO_MODEL_ANIMAL", "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/animal_detection.pt")
MODEL_PATH_HAND   = os.getenv("YOLO_MODEL_HAND",   "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/hand_detection.pt")

app = FastAPI(title="NON Realtime Detector — Dual Models Always")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ระบุโดเมนจริงในโปรดักชัน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== YOLO ======
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"ต้องติดตั้ง ultralytics ก่อน: {e}")

def _ensure(p: str, tag: str):
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"{tag} ไม่พบไฟล์โมเดล: {p}")

_ensure(MODEL_PATH_ANIMAL, "animal")
_ensure(MODEL_PATH_HAND,   "hand")

model_animal = YOLO(MODEL_PATH_ANIMAL)
model_hand   = YOLO(MODEL_PATH_HAND)
names_animal = model_animal.names
names_hand   = model_hand.names

# คิวเฟรม: กันไม่ให้เฟรมทับกันขณะประมวลผล (ลดดีเลย์)
infer_lock = asyncio.Semaphore(1)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # รัน 2 โมเดลคู่ขนาน

@app.get("/health")
def health():
    return {
        "ok": True,
        "animal": os.path.basename(MODEL_PATH_ANIMAL),
        "hand": os.path.basename(MODEL_PATH_HAND),
    }

def decode_image_from_bytes(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("decode image ล้มเหลว")
    return img  # BGR

def yolo_results_to_dets(results, names: Dict[int, str]) -> List[Dict,]:
    dets: List[Dict] = []
    res = results[0]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xywh.cpu().numpy()   # center x,y,w,h
        confs = res.boxes.conf.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy().astype(int)
        for (x, y, w, h), cf, c in zip(boxes, confs, clss):
            dets.append({
                "x": float(x), "y": float(y), "w": float(w), "h": float(h),
                "conf": float(cf), "cls": int(c),
                "label": str(names.get(int(c), str(int(c)))),
            })
    return dets

def run_predict_pair(img_bgr: np.ndarray):
    # รัน 2 โมเดลคู่ขนานบน image เดียวกัน
    def _pred(model):
        return model.predict(source=img_bgr, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

    fut1 = executor.submit(_pred, model_animal)
    fut2 = executor.submit(_pred, model_hand)
    r_animal = fut1.result()
    r_hand   = fut2.result()
    return r_animal, r_hand

def now_str():
    return time.strftime("%H:%M:%S")

@app.websocket("/ws")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data and data["bytes"] is not None:
                img_bytes = data["bytes"]
            else:
                txt = data.get("text")
                if not txt:
                    continue
                try:
                    if txt.startswith("data:image"):
                        b64 = txt.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64)
                    else:
                        img_bytes = base64.b64decode(txt)
                except Exception:
                    await websocket.send_text(json.dumps({"error": "invalid_payload"}))
                    continue

            try:
                img = decode_image_from_bytes(img_bytes)
            except Exception:
                await websocket.send_text(json.dumps({"error": "decode_failed"}))
                continue

            async with infer_lock:
                # รัน 2 โมเดลเสมอ
                r_animal, r_hand = run_predict_pair(img)
                det_animal = yolo_results_to_dets(r_animal, names_animal)
                det_hand   = yolo_results_to_dets(r_hand,   names_hand)

                # log ในเทอร์มินัล
                print(f"[{now_str()}] DUAL: animal={len(det_animal)}  hand={len(det_hand)}")
                for tag, arr in (("animal", det_animal), ("hand", det_hand)):
                    for i, d in enumerate(arr):
                        print(f"  [{tag}] #{i+1} {d['label']} {d['conf']:.2f} xywh=({d['x']:.1f},{d['y']:.1f},{d['w']:.1f},{d['h']:.1f})")

                payload = {
                    "ts": time.time(),
                    "mode": "dual",
                    "models": [
                        {"name": "animal_detection", "detections": det_animal},
                        {"name": "hand_detection",   "detections": det_hand},
                    ],
                }

            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
