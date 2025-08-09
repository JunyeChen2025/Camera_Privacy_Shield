# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import subprocess, cv2, numpy as np, time
from ultralytics import YOLO

# Parameters
WIDTH, HEIGHT = 640, 480      # Adjust as required
MODEL_PATH = "best.pt"                 # weight file

app = Flask(__name__)
model = YOLO(MODEL_PATH)

def mjpeg_frames(width=WIDTH, height=HEIGHT):
    cmd = [
        "rpicam-vid",
        "-t", "0",                       # Unlimited duration
        "--codec", "mjpeg",
        "--width", str(width),
        "--height", str(height),
        "-o", "-"                        # èOutput to stdout
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk

            # Search JPEG SOI/EOI
            start = buf.find(b"\xff\xd8")
            end   = buf.find(b"\xff\xd9")

            if start != -1 and end != -1 and end > start:
                jpg = buf[start:end+2]
                buf = buf[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img
    finally:
        proc.terminate()
        proc.wait()

def generate_frames():
    #Read MJPEG frames -> YOLO inference -> Output multipart MJPEG frame by frame
    last_annot = None
    i = 0
    for frame in mjpeg_frames():
        i += 1

        # Delay and load ruduction: inference can be done on every N frames, resuing previous results on other frames
        RUN_EVERY = 2            # Infer every two frames
        if i % RUN_EVERY == 0 or last_annot is None:
            # YOLOv8 æinference
            results = model(frame, imgsz=320, verbose=False)
            annot = results[0].plot()    # Plot the detection box
            last_annot = annot
        else:
            annot = last_annot

        # Encode JPEG
        ok, jpeg = cv2.imencode(".jpg", annot, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue

        data = jpeg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")   # Visualization on a webpage

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    try:
        subprocess.call(["pkill", "-f", "rpicam-vid"])
    except Exception:
        pass

    # Listen on 0.0.0.0:5000, Access browser http://192.168.0.123:5000
    app.run(host="0.0.0.0", port=5000, threaded=True)
