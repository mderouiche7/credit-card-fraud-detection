# Credit Card Fraud Detection Dashboard

[![HuggingFace Spaces](https://img.shields.io/badge/Gradio-App-blue?logo=gradio)](https://huggingface.co/spaces/xkakashi/credit-card-fraud-app)



An interactive fraud detection web app powered by **XGBoost** and **SHAP**, allowing users to input transaction data and receive real-time fraud risk scoring and transparent feature attributions through an intuitive Gradio interface.

---

## Project Overview

This project demonstrates a full **ML pipeline** for credit card fraud detection:
- Data exploration & preprocessing
- Model training with XGBoost and hyperparameter tuning
- Explainability using SHAP
- Scaler persistence
- Gradio-powered dashboard with prediction and SHAP visualization
- Batch mode CSV support

It aims to support financial institutions or analysts in identifying fraud with high precision while maintaining interpretability.

---

## Features

- **Single prediction UI:** Input transaction data manually via sliders and receive fraud probability with SHAP bar explanation.
- **Batch prediction UI:** Upload CSV files to detect fraud at scale.
- **Robust scaling:** Time and Amount are scaled consistently using a pre-trained StandardScaler.
- **SHAP integration:** Visual insights into which features influenced the decision.
- **Dot-based decimal validation:** Ensures correct input formatting.


---


### Inspired By:

This project is inspired by:

- **Practical Handbook**: *Reproducible Machine Learning for Credit Card Fraud Detection – A Practical Handbook*  
  [GitHub](https://github.com/username/project-link) <!-- Replace with real link -->

- **Doctoral Thesis**: *Dal Pozzolo, Andrea. "Adaptive Machine Learning for Credit Card Fraud Detection"*  
  [PDF](https://example.com/thesis.pdf) <!-- Replace with real link -->

---

### Author

**Mohamed Derouiche**  

---

### License

This project is open-sourced under the **MIT License**.


