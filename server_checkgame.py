# server_checkgame.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI + WebSocket (port 8003)
รับแพ็กเก็ตแบบ [uint32 header_len][header JSON {t0}][JPEG bytes]
ประมวลผลด้วยโมเดล YOLO .pt (game_detection.pt) แล้วส่งกลับผลลัพธ์เป็น JSON:
{
  "boxes": [{"x":..., "y":..., "w":..., "h":..., "label":..., "conf":...}],
  "w": src_width,
  "h": src_height,
  "t_ms": inference_ms,
  "t_ms_total": total_ms,
  "client_latency_ms": (ถ้า header.t0 มีค่า)
}

เงื่อนไขเพิ่มเติม:
- จะ "ส่งค่าไป" เฉพาะเมื่อมีผลตรวจจับที่มีความมั่นใจ (confidence) >= GAME_CONF_THRESHOLD
  (ค่าเริ่มต้น 0.7 หรือ 70%) มิเช่นนั้นจะข้ามการส่งข้อความในรอบนั้นไปเลย
"""

import os, json, time, struct
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# ===== โหลดโมเดล YOLO (Ultralytics) =====
# ติดตั้ง: pip install ultralytics fastapi uvicorn[standard] opencv-python-headless
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ไม่พบ ultralytics: โปรดติดตั้งด้วย 'pip install ultralytics'") from e

MODEL_PATH = os.environ.get(
    "GAME_MODEL_PATH",
    "/Users/sutinan/Desktop/Project/NON-AUTOS-MINE(Overall)/detection_model/model.pt/game_detection.pt",
)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")

# ค่า threshold สำหรับความมั่นใจ (ปรับได้ด้วย env: GAME_CONF_THRESHOLD)
CONF_THRESHOLD = float(os.environ.get("GAME_CONF_THRESHOLD", "0.7"))

model = YOLO(MODEL_PATH)

# ===== FastAPI app =====
app = FastAPI(title="CheckGame WS @8003", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=PlainTextResponse)
def index():
    return (
        "CheckGame WebSocket is running on /ws (binary JPEG + header JSON). "
        f"CONF_THRESHOLD={CONF_THRESHOLD}"
    )


# ===== Utilities =====
def decode_packet(data: bytes) -> Tuple[dict, bytes]:
    """แยก header(JSON) และ JPEG image bytes ออกจากแพ็กเก็ต"""
    if len(data) < 4:
        raise ValueError("packet too short")
    (hdr_len,) = struct.unpack("<I", data[:4])  # little-endian uint32
    if 4 + hdr_len > len(data):
        raise ValueError("invalid header length")
    hdr_bytes = data[4:4+hdr_len]
    jpeg_bytes = data[4+hdr_len:]
    header = json.loads(hdr_bytes.decode("utf-8")) if hdr_len else {}
    return header, jpeg_bytes


def jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode JPEG")
    return img


def run_inference_bboxes(img_bgr: np.ndarray) -> Tuple[List[dict], float]:
    """รัน YOLO แล้วคืนค่า list ของ boxes (xywh) พร้อมเวลา inference (ms)"""
    t0 = time.perf_counter()
    # ใช้ predict() เพื่อควบคุม verbose; รองรับโมเดลตรวจจับของ Ultralytics
    results = model.predict(img_bgr, verbose=False)[0]
    t1 = time.perf_counter()

    boxes_out: List[dict] = []
    names = results.names  # class id → label

    if hasattr(results, "boxes") and results.boxes is not None:
        # YOLOv8 format
        for b in results.boxes:
            # xyxy
            xyxy = b.xyxy[0].tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cls_id = int(b.cls[0]) if b.cls is not None else -1
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            label = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else (names[cls_id] if 0 <= cls_id < len(names) else str(cls_id))
            )
            boxes_out.append(
                {"x": x1, "y": y1, "w": w, "h": h, "label": label, "conf": conf}
            )

    infer_ms = (t1 - t0) * 1000.0
    return boxes_out, infer_ms


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = await ws.receive_bytes()
        except Exception:
            break
        try:
            t_total0 = time.perf_counter()

            # ---------- decode ----------
            header, jpeg_bytes = decode_packet(data)
            t0_client = float(header.get("t0", 0)) if isinstance(header, dict) else 0.0

            # ---------- decode JPEG ----------
            img = jpeg_to_bgr(jpeg_bytes)
            h, w = img.shape[:2]

            # ---------- YOLO inference ----------
            boxes_all, t_infer = run_inference_bboxes(img)

            # ---------- filter by confidence ----------
            boxes_confident = [b for b in boxes_all if b.get("conf", 0.0) >= CONF_THRESHOLD]

            # metric
            t_total1 = time.perf_counter()
            total_ms = (t_total1 - t_total0) * 1000.0
            client_latency_ms = None
            if t0_client:
                # ฝั่ง client ส่ง t0 = performance.now() (ms);
                # ในที่นี้เปลี่ยน scale ให้ใกล้เคียง โดยเทียบ time.perf_counter()*1000
                now_ms = time.perf_counter() * 1000.0
                client_latency_ms = max(0.0, now_ms - t0_client)

            # ---------- ส่งผลลัพธ์เฉพาะเมื่อมั่นใจ >= threshold ----------
            if boxes_confident:
                payload = {
                    "boxes": boxes_confident,
                    "w": w,
                    "h": h,
                    "t_ms": t_infer,
                    "t_ms_total": total_ms,
                    "conf_threshold": CONF_THRESHOLD,
                }
                if client_latency_ms is not None:
                    payload["client_latency_ms"] = client_latency_ms

                await ws.send_text(json.dumps(payload))
            # else: ไม่ส่งอะไรในรอบนี้ ตามข้อกำหนด "เมื่อมีความมั่นใจมากกว่า 70% ถึงจะทำการส่งค่าไป"

        except Exception as e:
            # ส่งข้อผิดพลาดกลับ (ช่วยดีบักที่หน้างาน)
            try:
                await ws.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    # หมายเหตุ: ใช้ชื่อโมดูลไฟล์นี้ (server_checkgame:app) เมื่อรันแบบโมดูล
    uvicorn.run("server_checkgame:app", host="0.0.0.0", port=port, reload=False)
