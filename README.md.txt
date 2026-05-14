# 🏠 House Price Prediction using MLP (PyTorch)

## 📌 Project Overview
This project predicts house prices using a Multilayer Perceptron (MLP) neural network built with PyTorch.  
The dataset used is the Ames Housing dataset.

---

## 📊 Dataset
- Source: Kaggle Ames Housing Dataset  
- Link: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

---

## ⚙️ Workflow
- Data preprocessing
- Missing value handling
- One-hot encoding
- Feature scaling
- Log transformation of target
- Training MLP model

---

## 🧠 Model Architecture
- Input Layer
- Hidden Layers: 512 → 256 → 128
- Activation: ReLU
- Regularization: Dropout + BatchNorm
- Loss Function: Smooth L1 Loss (Huber Loss)

---

## 📈 Evaluation
- Metric: Mean Squared Error (MSE)
- Visualization: Actual vs Predicted prices

---

## 📊 Result
Final Test MSE: ~7.6e8

---

## 🚀 Future Improvements
- Feature selection to reduce noise
- Use XGBoost / LightGBM
- Hyperparameter tuning
- K-Fold Cross Validation
- Better categorical encoding

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python model.py