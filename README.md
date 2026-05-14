# 🏠 House Price Prediction using MLP (PyTorch)

---

## 🧠 Project Overview

This project implements a **House Price Prediction system** using a **Multilayer Perceptron (MLP)** built with PyTorch.

The goal is to predict house sale prices based on structured tabular data from the **Ames Housing dataset**.

This is a regression problem where the model learns non-linear relationships between house features and final price.

---

## 🎯 Problem Statement

Predict the final sale price of residential homes using features such as:

- House size and area
- Quality of construction
- Garage information
- Year built / remodel year
- Neighborhood and other categorical attributes

---

## 📊 Dataset

- Dataset: Ames Housing Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset  
- Samples: ~2930 rows  
- Features: 80+ features  
- Target: `SalePrice`

---

## ⚙️ Data Preprocessing

### 1. Missing Values Handling
- Numerical features → Mean Imputation  
- Categorical features → Most Frequent Value  

### 2. Encoding
- One-Hot Encoding for categorical variables  

### 3. Feature Scaling
- StandardScaler applied to numerical features  

### 4. Target Transformation
To reduce skewness:

```python
y = log(1 + y)
5. Train/Test Split
80% training
20% testing
Random state = 42
🧠 Model Architecture (MLP)
Input Layer
↓
Linear (512) → BatchNorm → ReLU → Dropout
↓
Linear (256) → BatchNorm → ReLU → Dropout
↓
Linear (128) → ReLU
↓
Output Layer (1 neuron)
Activation Function:
ReLU
Regularization:
Dropout
Batch Normalization
⚙️ Training Details
Epochs: 150
Batch Size: 64
Optimizer: Adam
Learning Rate: 0.0005
Weight Decay: 1e-4
Loss Function: Smooth L1 Loss (Huber Loss)
📈 Evaluation Metric
Mean Squared Error (MSE)
MSE = (1/n) * Σ (y - ŷ)^2
📊 Results
Final Test MSE: ~7.6 × 10^8
📉 Observations
Model captures general trends in housing prices
Higher error on expensive houses
Dataset is highly non-linear
One-hot encoding increases dimensionality significantly
MLP is sensitive to noisy features in tabular data
🚀 Future Improvements
🔹 Feature Engineering
Feature selection to remove irrelevant features
Correlation analysis
Reduce noise from high-dimensional features
🔹 Encoding Improvements
Replace One-Hot Encoding with:
Target Encoding
Embedding Layers
🔹 Advanced Models
XGBoost
LightGBM
CatBoost
🔹 Model Optimization
Hyperparameter tuning
Grid Search / Optuna
🔹 Cross Validation
K-Fold Cross Validation for stable evaluation
🔹 Dimensionality Reduction
PCA for feature compression
🔹 Regularization
L1 / L2 regularization
Adjust dropout rates
📁 Project Structure
House-Price-MLP/
│
├── model.py / model.ipynb
├── requirements.txt
├── README.md
├── AmesHousing.csv (optional)
└── results/
    └── prediction_plot.png
💻 How to Run
pip install -r requirements.txt
python model.py
🛠️ Technologies Used
Python
PyTorch
Pandas
NumPy
Scikit-learn
Matplotlib
👨‍💻 Conclusion

This project demonstrates a full deep learning pipeline for regression using an MLP model.

It shows how neural networks can learn complex patterns in tabular data, while also highlighting limitations compared to tree-based models.
