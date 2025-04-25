# Homework 3 – Tracking  
**Course**: Video Streaming and Tracking  
**Semester**: 2023 Fall  
**Deadline**: 2023/12/10 23:59

---

## 🎯 Objective

- Implement **visual multiple object tracking** on videos
- Use **detection model + Hungarian algorithm** for tracking
- Count the total number of people who appeared in the video
- Output:
  - A video with tracking results (drawn bounding boxes)
  - Final count printed on terminal (e.g., `count: 5`)

---

## 📦 Steps

1. **Detection**  
   - Choose a detection model (YOLO, Faster R-CNN, etc.)
   - Detect the **human** category in each frame
   - You can use **pre-trained models** or train your own

2. **Tracking**  
   - Use the **Hungarian Algorithm** to match bounding boxes across frames
   - Decide on your **cost function** (e.g., IoU, distance, size, ReID, etc.)
   - Assign unique **colors** for different objects
   - A person leaving and re-entering = different ID

3. **Output**  
   - Show tracking results in a video
   - Print the total number of people on the **terminal** (no text file needed)

---

## 🧪 Benchmark Data

- Provided:
  - `easy_9.mp4` – Simple scene (no overlaps)
  - `hard_9.mp4` – Complex scene (with overlaps)
- File naming: the number in filename = correct count (e.g., `easy_9.mp4` → 9 people)
- You can also test on your own videos

---

## 🧮 Grading Policy

### 📹 Model Implementation – 50 pts
- Show result video of `easy_9.mp4`
- Track with different colors per object
- ❗ Do **not** use tracking models like DeepSORT (deduct -20 pts)

### 🔢 Counting Accuracy
- Easy Case – 10 pts
  - Output count must match ground truth of `easy_9.mp4`
- Hard Case (demo.mp4) – 20 pts
  - Accuracy based on how close your count is:
    - ±3 → 20 pts
    - ±6 → 15 pts
    - ±9 → 10 pts
    - Beyond → partial score

### 🗣️ Q&A – 20 pts
- TA will ask implementation-related questions during demo

---

## 📅 Demo

- Demo via **Google Meet** on:
  - 12/11、12/14、12/15
- Demo時間填寫表單（11/20 開放填寫）：
  [Demo Sheet Link](https://docs.google.com/spreadsheets/d/1JXhLFaYfQdGRxpVAcRi2wPsz50cGW9oDUNLhmBSfZr4/edit?usp=sharing)

---

## 📤 Hand-in Rules

- Submit a single zip file: `HW3_[studentID].zip`
- The zip must contain:
  - ✅ Code only
  - ❌ Do not include model weights or video files

---

## ⚠️ Penalties

| 類型           | 扣分方式           |
|----------------|--------------------|
| 命名／格式錯誤 | -10 pts             |
| 遲交           | 每天 -20%          |
| 抄襲他人       | 直接 0 分           |

---

## 🔗 References

- 匈牙利演算法 Wiki: https://zh.wikipedia.org/zh-tw/匈牙利算法  
- 匈牙利演算法簡介: https://hackmd.io/@computerVision/S18nD20Vq  
- Edge AI Tracking 實作: https://alu2019.home.blog/2021/01/20/edge-ai-multiple-object-tracking-mot-duo-ge-wu-ti/  
- Tracking 演算法講解: https://blog.csdn.net/your_answer/article/details/79160045

---
