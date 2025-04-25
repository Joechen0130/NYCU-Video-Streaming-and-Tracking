# Homework 1 - Video Streaming and Tracking

This assignment is designed to help you practice building a neural network from scratch using **PyTorch** to perform a **classification task** on a sports image dataset. Pretrained weights and existing models (e.g., torchvision models) **are not allowed**.

---

## 📌 Objectives

- Implement a custom neural network in PyTorch
- Train and evaluate the network on a 12-class sports image dataset
- Ensure full reproducibility of your code
- Optimize model performance and parameter size

---

## 📝 Task Description

You must complete and submit the following Python files:

- `net.py`: Define your network class named `my_network`
- `train.py`: Train your model
- `test.py`: Test your model and generate prediction CSV

📌 **Restrictions**:
- You **must not** use pretrained models or pretrained weights.
- You **may** import any PyTorch packages and write helper functions.

---

## 📂 Dataset

- **Expires**: Link valid until HW1 deadline
- **Description**:
  - 12 categories of sports images
  - Image size: `224 x 224 x 3`
  - Training set: 1606 images
  - Test set: 120 images
  - ⚠️ No validation set provided — split it yourself from training set

---

## 🏁 How to Run

1. Set up the environment from `env.yml`
2. Place the dataset in your project directory
3. Run the test script:
   ```bash
   python test.py

Your output should be a CSV file named:
    pred_{student_id}.csv

## 🧮 Grading

### 🎯 Top-1 Accuracy (80 points)
| Accuracy Range         | Points Awarded |
|------------------------|----------------|
| ≥ 55%                  | 80 pts         |
| 50% – 54%              | 70 pts         |
| 35% – 49%              | 60 pts         |
| < 35%                  | 0 pts          |

### 📉 Parameter Count (20 points)
| Ranking (by fewest parameters) | Points Awarded |
|--------------------------------|----------------|
| Top 20%                        | 20 pts         |
| 20% – 40%                      | 16 pts         |
| 40% – 60%                      | 12 pts         |
| 60% – 80%                      | 8 pts          |
| Bottom 20%                     | 4 pts          |
