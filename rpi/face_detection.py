# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

#Initialize camera in OpenCV
#cap = None
#for i in range(5):
#	cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
#	if cap.isOpened():
#		print(f"Camera index {i} is available")
#		break
#if not cap or not cap.isOpened():
#	raise RuntimeError("Cant open any cameras")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Loading model
model = YOLO("best.pt")

print("Ready to capture camera frames using rpicam-jpeg")

def generate_frames():
#	frame_count = 0
#	while cap.isOpened():
#		ret, frame = cap.read()
#		if not ret:
#			print("Fail reading frame!")
#			cap.release()
#			cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
#			continue

#		print(f"Processing frame {frame_count}", end='\r')
#		frame_count += 1
	import subprocess
	import os
	import time

	while True:
		try:
			if os.path.exists("face.jpg"):
				os.remove("face.jpg")

			subprocess.run(["rpicam-jpeg", "-o",  "face.jpg", "-t", "100", "-n"], check=True)

			time.sleep(0.05)

			if not os.path.exists("face.jpg"):
				continue

			frame = cv2.imread("face.jpg")
			if frame is None:
				print("Failed to read image")
				continue


			#YOLOv8 inference
			results = model(frame, imgsz=320, verbose=False)
			annotated_frame = results[0].plot()

			#Encoding JPEG
			ret, buffer = cv2.imencode('.jpg', annotated_frame)
			if not ret:
				continue

			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
		except Exception as e:
			print(f"[ERROR] {e}")
			continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()
