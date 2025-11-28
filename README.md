# 💳 FP_FraudDetector_IAA

## Credit Card Fraud Detection using Machine Learning

This project implements a **Deep Learning** model (Neural Network) to classify financial transactions as either fraudulent or legitimate. It specifically addresses the crucial challenge of **severe class imbalance** inherent in real-world fraud detection datasets.

The analysis is based on a credit card transaction dataset sourced from **Kaggle**, where sensitive features were transformed using **Principal Component Analysis (PCA)** to ensure user privacy.

---

## 🚀 Project Structure

The repository follows a standard data science project structure to ensure reproducibility, clear separation of concerns, and ease of navigation.

FP_FraudDetector_IAA/
├── data/
│   ├── raw/                # Original, immutable raw data (Kaggle dataset).
│   └── processed/          # Cleaned, scaled, and balanced data used for modeling.
├── notebooks/
│   └── Detecting_Credit_Card_Fraud.ipynb   # EDA and modeling logic.
├── src/                    # Python scripts for reusable functions (e.g., custom transformers).
├── models/                 # Trained and serialized models (.h5, .pkl, etc.).
└── README.md               # Project documentation and entry point.

---

## 💾 Setup and Installation

To replicate this analysis environment, it is highly recommended to use a virtual environment and install the required dependencies.

* **Python Version:** 3.x
* **Key Libraries:** `pandas`, `numpy`, `scikit-learn`, `tensorflow`/`keras`, `imblearn`.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate




