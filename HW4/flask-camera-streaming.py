#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, request, jsonify
import cv2
from Tracking import *
app = Flask(__name__)

# Model
#https://github.com/ultralytics/yolov5#pretrained-checkpoints
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
model.conf = 0.6  # NMS confidence threshold
model.iou = 0.1  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#model.max_det = 1000  # maximum number of detections per image
model.amp = False

video_path = r"./demo_test.mp4"
cap = cv2.VideoCapture(0)

fps = int(cap.get(5))
print('fps:', fps)
t = int(1000/fps)

tracker=Sort(max_age=30,min_hits=10,iou_threshold=0.1)

def gen_frames():
    while True:
        detections=np.empty((0,6))
        success, frame = cap.read()
        if not success:
            break
        else:
            # Inference
            results = model(frame[..., ::-1])
            #print(results.pandas().xyxy[0])
            for index, row in results.pandas().xyxy[0].iterrows():
                xmin,ymin,xmax,ymax,confidence,class_name = int(row["xmin"]),int(row["ymin"]),int(row["xmax"]),int(row["ymax"]),row["confidence"],row["name"]
                currentArray=np.array([xmin,ymin,xmax,ymax,confidence,0])
                detections=np.vstack((detections,currentArray))
        
            resultTracker=tracker.update(detections)
            for result in resultTracker:
                x1, y1, x2, y2 ,id_ = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 6)
                cx,cy=x1+w//2,y1+h//2

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_mouse_position', methods=['POST'])
def get_mouse_position():
    data = request.get_json()
    print(data)
    resp = jsonify(success=True)
    return resp

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0')