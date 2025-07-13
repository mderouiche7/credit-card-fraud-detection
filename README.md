# 💳 Credit Card Fraud Detection Dashboard

[![HuggingFace Spaces](https://img.shields.io/badge/Gradio-App-blue?logo=gradio)](https://huggingface.co/spaces/mderouiche7/credit-card-fraud-app)

An interactive fraud detection web app powered by **XGBoost** and **SHAP**, allowing users to input transaction data and receive real-time fraud risk scoring and transparent feature attributions through an intuitive Gradio interface.

---

## 🎓 Project Overview

This project demonstrates a full **ML pipeline** for credit card fraud detection:
- Data exploration & preprocessing
- Model training with XGBoost and hyperparameter tuning
- Explainability using SHAP
- Scaler persistence
- Gradio-powered dashboard with prediction and SHAP visualization
- Batch mode CSV support

It aims to support financial institutions or analysts in identifying fraud with high precision while maintaining interpretability.

---

## 📅 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Rate:** ~0.17%
- **Features:** `V1-V28` (PCA anonymized), `Time`, `Amount`, `Class`

---

## ⚙️ Technologies Used

- Python 3.11+
- XGBoost
- SHAP
- scikit-learn
- Pandas, NumPy, Matplotlib
- Gradio
- Hugging Face Spaces
- Google Colab (for dev & training)

---

## 🌟 Features

- **Single prediction UI:** Input transaction data manually via sliders and receive fraud probability with SHAP bar explanation.
- **Batch prediction UI:** Upload CSV files to detect fraud at scale.
- **Robust scaling:** Time and Amount are scaled consistently using a pre-trained StandardScaler.
- **SHAP integration:** Visual insights into which features influenced the decision.
- **Dot-based decimal validation:** Ensures correct input formatting.

---

## 🌐 Live Demo

> **Try it live on Hugging Face Spaces**:
>
🔴 Live App: [Launch on Hugging Face](https://huggingface.co/spaces/xkakashi/credit-card-fraud-app)

---
🎡 Setup Instructions

# Clone the repo
$ git clone https://github.com/mderouiche7/credit-card-fraud-detection
$ cd credit-card-fraud-detection

# Install dependencies
$ pip install -r requirements.txt

# Run the app (optional if not using HF Space)
$ python app.py


### 🚀 Run the App (Optional if not using Hugging Face Space)

```bash
python app.py
```

---

### 🧠 Based On

This project is inspired by:

- 📘 **Practical Handbook**: *Reproducible Machine Learning for Credit Card Fraud Detection – A Practical Handbook*  
  [GitHub](https://github.com/username/project-link) <!-- Replace with real link -->

- 📄 **Doctoral Thesis**: *Dal Pozzolo, Andrea. "Adaptive Machine Learning for Credit Card Fraud Detection"*  
  [PDF](https://example.com/thesis.pdf) <!-- Replace with real link -->

---

### 👨‍💻 Author

**Mohamed Derouiche**  
[GitHub](https://github.com/mderouiche7) • [LinkedIn](https://www.linkedin.com/in/mohamed-derouiche-ba1843294)

---

### 🤝 Want to Contribute?

Feel free to open issues or pull requests if you'd like to:

- Add more models (e.g., LightGBM, Neural Nets)
- Add Streamlit or FastAPI support
- Improve visualizations (e.g., Plotly dashboards)
- Enable real-time detection APIs

---

### 🔐 License

This project is open-sourced under the **MIT License**.


