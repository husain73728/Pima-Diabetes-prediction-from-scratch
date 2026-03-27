# 🧠 Diabetes Prediction using Neural Network (From Scratch)

## 📌 Overview
This project implements a fully connected **Artificial Neural Network (ANN)** from scratch using **NumPy** to predict diabetes risk based on medical features.

The goal of this project is to demonstrate a deep understanding of:
- Forward propagation
- Backpropagation
- Gradient descent
- Loss functions

without relying on high-level libraries like TensorFlow or PyTorch.

---

## 🚀 Features
- Neural network built entirely from scratch
- Two hidden layers with ReLU activation
- Sigmoid output for binary classification
- Binary Cross-Entropy loss
- Manual gradient computation and weight updates
- Accuracy evaluation

---

## 🧱 Model Architecture
| Layer            | Size | Activation |
|------------------|------|------------|
| Input Layer      | 8    | —          |
| Hidden Layer 1   | 16   | ReLU       |
| Hidden Layer 2   | 8    | ReLU       |
| Output Layer     | 1    | Sigmoid    |

---

## 📊 Dataset
The model uses the **Pima Indians Diabetes Dataset**, consisting of 8 medical features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target:**
- `0` → Non-diabetic  
- `1` → Diabetic  

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/diabetes-ann-from-scratch.git
cd diabetes-ann-from-scratch
pip install numpy
