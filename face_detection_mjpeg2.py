# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import subprocess, cv2, numpy as np, time, threading, os
from ultralytics import YOLO
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets
import binascii

# Parameters
WIDTH, HEIGHT = 480, 360         # Lower resolution to shorten queue
MODEL_PATH = "best.pt"
RUN_EVERY = 2                             # Infer for every N frames次
JPEG_QUALITY = 65                         # Lower quality to reduce latency

#Secret Key Access
KEY_PATH = os.path.expanduser("~/.camera_secret.key")
with open(KEY_PATH, "rb") as f:
    key_hex = f.read().strip()
AES_KEY = binascii.unhexlify(key_hex)
assert len(AES_KEY) in (16, 24, 32), "AES key must be 16/24/32 bytes"

#AES-CTR Encryption Function
def encrypt_roi_aes_ctr(img, box, key=AES_KEY):
    #Encrypts the pixels inside the frames and writes it back to image
    x1, y1, x2, y2 = [int(v) for v in box]
    #Boundary clipping
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return

    iv = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
    encryptor = cipher.encryptor()

    roi_c = np.ascontiguousarray(roi)
    enc_bytes = encryptor.update(roi_c.tobytes()) + encryptor.finalize()

    roi[:, :, :] = np.frombuffer(enc_bytes, dtype=np.uint8).reshape(roi.shape)

# Initialization
app = Flask(__name__)
model = YOLO(MODEL_PATH)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Share the latest frames
_latest = None
_lock = threading.Lock()

def reader():
    #Independent thread: Parse MJPEG from rpicam-vid to retain the latest frames
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--codec", "mjpeg",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--quality", str(JPEG_QUALITY),
        "-o", "-"
    ]

    cmd.insert(1, "--nopreview")
    cmd.append("--flush")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk: break
            buf += chunk

            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpeg = buf[start:end+2]
                    buf = buf[end+2:]
                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with _lock:
                            _latest = globals()["_latest"]
                            globals()["_latest"] = img
                    else:
                         pass
                else:
                    break
    finally:
        try: proc.terminate()
       	except: pass
        try: proc.wait()
        except: pass

# Preheat to reduce jitters
_ = model.predict(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), imgsz=256, verbose=False)

def generate_frames():
    #Access the latest frame -> yolo infernece for every N frames -> Encode JPEG -> MJPEG output
    #Enable thread capturing frame
        if not any(isinstance(t, threading.Thread) and t.name == "reader"
                   for t in threading.enumerate()):
            t = threading.Thread(target=reader, name = "reader", daemon=True)
            t.start()

        last_boxes = None
        i = 0

        while True:
            with _lock:
                frame = None if globals()["_latest"] is None else globals()["_latest"].copy()

            if frame is None:
                time.sleep(0.005)
                continue

            i += 1

            try:
               if i % RUN_EVERY == 0 or last_boxes is None:
                   #Smaller inference size but faster speed
                   results = model(frame, imgsz=256, verbose=False)
                   #last_annot = results[0].plot()
                   boxes_tensor = results[0].boxes.xyxy if hasattr(results[0].boxes, "xyxy") else None
                   if boxes_tensor is not None and len(boxes_tensor) > 0:
                       last_boxes = boxes_tensor.cpu().numpy().astype(int)
                   else:
                       last_boxes = np.empty((0, 4), dtype=int)

               boxes = last_boxes if last_boxes is not None else np.empty((0, 4), dtype=int)

            #Encrypt every ROI usning AES_CTR on copies of frames
               for b in boxes:
                    encrypt_roi_aes_ctr(frame, b, key=AES_KEY)
                    x1,y1,x2,y2 = [int(v) for v in boxes]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                    cv2.putText(frame, "encrypted", (x1, max(0,y1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                #annot = last_annot

            except Exception as e:
                print("Pipeline error:", e)
                pass

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue

            data = jpeg.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    try: subprocess.call(["pkill", "-f", "rpicam-vid"])
    except Exception: pass
    app.run(host="0.0.0.0", port=5000, threaded=True)
