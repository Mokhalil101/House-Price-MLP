📘 House Price Prediction using MLP (PyTorch)

🧠 1. Project Overview

This project implements a House Price Prediction system using a Multilayer Perceptron (MLP) built with PyTorch.

The goal is to predict house sale prices based on structured tabular data from the Ames Housing dataset.

This is a regression problem where the model learns complex non-linear relationships between house features and their final prices.

🎯 2. Problem Statement

The objective is to predict the final sale price of residential homes using multiple features such as:

House size and area
Quality of construction and materials
Garage attributes
Year built and remodel year
Neighborhood and categorical features
📊 3. Dataset Description
Dataset: Ames Housing Dataset
Source: Kaggle
Samples: ~2930 rows
Features: 80+ features
Target: SalePrice
Feature Types:
Numerical features (e.g., Lot Area, Garage Area)
Categorical features (e.g., Neighborhood, House Style)
⚙️ 4. Data Preprocessing Pipeline
4.1 Handling Missing Values
Numerical features → Mean Imputation
Categorical features → Most Frequent Value
4.2 Feature Encoding
One-Hot Encoding applied to categorical variables
4.3 Feature Scaling
StandardScaler applied to numerical features
4.4 Target Transformation

To reduce skewness in price distribution:

y = log(1 + y)
4.5 Train-Test Split
Training set: 80%
Testing set: 20%
Random state: 42
🧠 5. Model Architecture (MLP)

A fully connected neural network:

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
Dropout (0.2 – 0.35)
Batch Normalization
⚙️ 6. Loss Function & Optimizer
Loss Function:
Smooth L1 Loss (Huber Loss)
Why?
More robust to outliers than MSE
Optimizer:
Adam Optimizer
Learning Rate: 0.0005
Weight Decay: 1e-4
🏋️ 7. Training Process
Epochs: 150
Batch Size: 64
Training Steps:
Forward pass
Loss computation
Backpropagation
Parameter update
📈 8. Evaluation Metrics
Metric Used:
Mean Squared Error (MSE)
Formula:
MSE=
n
1
	​

∑(y−
y
^
	​

)
2
Final Result:
Final Test MSE: ~7.6 × 10^8
📊 9. Visualization

A scatter plot is used to compare:

X-axis: Actual Prices
Y-axis: Predicted Prices

This helps evaluate how close predictions are to real values.

📉 10. Key Observations
Model captures general trends in the data
Higher error on expensive houses
Dataset is highly non-linear
One-hot encoding increases feature dimensionality significantly
MLP is sensitive to noisy features in tabular data
🚀 11. Future Improvements

Although the current MLP model provides reasonable performance, several improvements can be applied:

🔹 Feature Engineering
Feature selection to remove irrelevant variables
Correlation analysis and importance ranking
Reduce noise from high-dimensional features
🔹 Encoding Improvements
Replace One-Hot Encoding with:
Target Encoding
Embedding layers
🔹 Advanced Models
XGBoost
LightGBM
CatBoost

These models are often more effective for tabular data.

🔹 Model Optimization
Hyperparameter tuning (learning rate, layers, batch size)
Automated tuning using Grid Search or Optuna
🔹 Cross Validation
K-Fold Cross Validation
Improves stability and reduces variance
🔹 Dimensionality Reduction
PCA (Principal Component Analysis)
Reduces feature space while preserving variance
🔹 Regularization Enhancements
L1 / L2 regularization
Higher dropout rates
📁 12. Project Structure
House-Price-MLP/
│
├── model.py / model.ipynb
├── requirements.txt
├── README.md
├── AmesHousing.csv (optional)
└── results/
    └── prediction_plot.png
💻 13. How to Run the Project
pip install -r requirements.txt
python model.py
🛠️ 14. Technologies Used
Python
PyTorch
Pandas
NumPy
Scikit-learn
Matplotlib
🧾 15. Conclusion

This project demonstrates a complete deep learning pipeline for regression using an MLP model.

Despite being a simple architecture, it captures meaningful relationships in the dataset, but also highlights the limitations of neural networks for tabular data compared to gradient boosting methods.

🚀 16. Final Note

This project serves as a baseline deep learning regression model, with several planned improvements for future versions.
