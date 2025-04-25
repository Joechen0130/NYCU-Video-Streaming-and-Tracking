# HW4 - Flask Video Streaming with Object Tracking  

## 📌 Project Overview

This project extends the functionality of **Homework 3** by integrating:

- Real-time object tracking
- Video streaming through a Flask web server
- Interactive UI via JavaScript for controlling object visibility

---

## 🔧 System Architecture

### 🧠 Tracking System
- **Tracking Model**: Based on the tracking system from HW3
- **Detection**: YOLOv5 is used for object detection
- **Matching Logic**:
  1. `D > T`: Create new Kalman trackers for unmatched detections  
  2. `D == T`: Direct one-to-one matching, update trackers  
  3. `D < T`: Handle disappearing objects  

### 🎥 Frame Streaming
- Video loaded via `cv2.VideoCapture`
- Each frame is converted to JPEG format
- A Python `generator` is used to yield each frame to the client

### 🌐 Web Server
- Backend: **Flask**
- Frontend: **JavaScript + AJAX**
- Mouse click coordinates are captured on the video and sent back to the server via AJAX
- Python server checks whether the clicked point intersects with any tracking box:
  - If so, the corresponding object's `is_show` attribute is set to `False`

---

## 💡 Features

- Object detection and tracking across video frames
- Real-time video streaming to browser via Flask
- User can click on tracked objects to hide them from the stream
- Modular tracking logic using Kalman filters

---

## 🛠 Tech Stack

| Component      | Tool/Library           |
|----------------|------------------------|
| Object Detection | YOLOv5 (PyTorch)       |
| Tracking         | Kalman Filter          |
| Video Stream     | OpenCV + Flask         |
| Web Interaction  | JavaScript + AJAX      |

---

## 🚀 How to Run

1. 安裝依賴：
   ```bash
   pip install -r requirements.txt
2. 啟動 Flask server：
    ```bash
    python app.py
3. 開啟瀏覽器並進入：
    ```bash
    http://localhost:5000