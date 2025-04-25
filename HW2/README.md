# Homework 2 - Object Detection  
**Course**: Video Streaming and Tracking  
**Framework**: PyTorch  
**Model Base**: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

---

## 📌 Introduction

In this assignment, you will:

- Train a neural network to detect **cars** in a custom GTA dataset
- Use **YOLOX** as the baseline model
- Add **Squeeze-and-Excitation (SE)** module to enhance feature learning
- Compare performance with and without SE
- Evaluate your model using **mAP (mean Average Precision)**

---

## 🗂 Dataset

- Source: GTA video dataset
- **Classes**: Only detect cars (class = 0)
- **Training set**: 2039 images
- **Validation set**: 240 images
- **Testing set**: 720 images

### 🏷 Label Format (train/val):
Each label is a row:

### 📥 Dataset conversion:
You must convert this format into **YOLOX-compatible COCO format**.

---

## 🧪 Evaluation Metrics

- Use **PASCAL VOC 2012 mAP metric**
- Tool: [Object Detection Metrics GitHub Repo](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- IoU threshold = **0.85**
- Example usage:
```bash
python pascalvoc.py -t 0.85 -gtformat xyrb -detformat xyrb -np
