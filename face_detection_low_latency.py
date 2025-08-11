# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import subprocess, cv2, numpy as np, time, threading
from ultralytics import YOLO
import torch

# Adjustable parameters
WIDTH, HEIGHT = 480, 360         # Lower resolution / Add frame rate to shorten queue
MODEL_PATH = "best.pt"
RUN_EVERY = 2                             # Model infer for every N frames次
JPEG_QUALITY = 65                         # Lower latency

# Initialization
app = Flask(__name__)
model = YOLO(MODEL_PATH)
cv2.setNumThreads(1)
try:
    torch.set_num_threads(2)
except Exception:
    pass

# Share the latest frames
_latest = None
_lock = threading.Lock()

def reader():
    #Parse MJPEG from rpicam-vid stdout, only remain the latest frames --- independent thread
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
            start = buf.find(b"\xff\xd8")
            end   = buf.find(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = buf[start:end+2]
                buf = buf[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with _lock:
                        global _latest
                        _latest = img
    finally:
        proc.terminate()
        proc.wait()

# Preheat to reduce jitters on the first frame
_ = model.predict(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), imgsz=256, verbose=False)

def generate_frames():
    #Take latest frame -> (N frames apart) YOLO inference -> encode JPEG -> MJPEG output
    # Enable catching frames thread
    if not any(isinstance(t, threading.Thread) and t.name == "mjpeg_reader" for t in threading.enumerate()):
        t = threading.Thread(target=reader, name="mjpeg_reader", daemon=True)
        t.start()

    last_annot = None
    i = 0
    while True:
        with _lock:
            frame = None if _latest is None else _latest.copy()
        if frame is None:
            time.sleep(0.005)
            continue

        i += 1
        if i % RUN_EVERY == 0 or last_annot is None:
            # Smaller inference size, Faster speed�快
            results = model(frame, imgsz=256, verbose=False)
            last_annot = results[0].plot()
        annot = last_annot

        ok, jpeg = cv2.imencode(".jpg", annot, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
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
