import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
files.upload()  # upload scaler.pkl from your computer

#Uploasing model XGBoost best performer and the scaler:
model = joblib.load("/content/drive/MyDrive/models/xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

import kagglehub
# Download dataset from Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
# Load the dataset using the full path
df = pd.read_csv(f"{path}/creditcard.csv")

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split train/test (use same random_state as training for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Now, instead of fitting scaler again, just transform Time and Amount using loaded scaler:
X_test_scaled = X_test.copy()
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test_scaled[['Time', 'Amount']])

# Now X_test_scaled is ready for predictions or SHAP analysis

# Load model and scaler (adjust paths)

explainer = shap.TreeExplainer(model)

# Feature names (exclude target)
feature_names = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28", "Amount"
]

# Define UI inputs with sliders with step=0.01 for decimal precision
def get_inputs():
    return [
        gr.Slider(minimum=0, maximum=172792, step=1, value=50000, label="Time (seconds)"),
        *[gr.Slider(minimum=-30.0, maximum=30.0, step=0.01, value=0.0, label=feature) for feature in feature_names[1:-1]],
        gr.Slider(minimum=0.0, maximum=2500.0, step=0.01, value=50.0, label="Amount (Euros)")
    ]

# Prediction + SHAP function with input validation (accept dot decimals only)
def predict_with_shap(*inputs):
    try:
        # Convert inputs to float, replace comma with dot if any (reject commas by error)
        parsed_inputs = [float(str(x).replace(',', '.')) for x in inputs]
        X = np.array(parsed_inputs).reshape(1, -1)

        # Extract Time and Amount columns and scale using column names
        scaled_df = pd.DataFrame(X[:, [0, -1]], columns=['Time', 'Amount'])
        scaled_values = scaler.transform(scaled_df)

        # Put scaled values back in X
        X[:, 0] = scaled_values[:, 0]  # Scaled Time
        X[:, -1] = scaled_values[:, 1]  # Scaled Amount


        # Predict fraud probability
        pred_prob = model.predict_proba(X)[0][1]

        # Compute SHAP values
        shap_values = explainer.shap_values(X)

       # Plot SHAP bar summary
        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("shap_force_plot.png")
        plt.close()

        return f"Fraud Probability: {pred_prob:.4f}", "shap_force_plot.png"

    except ValueError:
        return "Error: Please enter decimals using dots (e.g. 6.5), not commas (6,5).", None
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", None

# Batch prediction for uploaded CSV
def batch_predict(file):
    try:
        if not file.name.endswith(".csv"):
            return "Error: Please upload a valid .csv file."

        df = pd.read_csv(file.name)
        X = df[feature_names]

        # Scale Time and Amount
        X[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])
        probs = model.predict_proba(X)[:, 1]
        df['Fraud_Probability'] = probs

        output_path = f"batch_predictions_{uuid.uuid4().hex[:6]}.csv"
        df.to_csv(output_path, index=False)
        return output_path

    except Exception as e:
        return f"Error processing file: {str(e)}"

# Footer and project links

description = """
Provide transaction features to estimate fraud probability using a pre-trained ML model.
SHAP explainability will highlight the most influential features.

**Input Guide:**
- `Time`: Seconds since the dataset's first transaction (range: 0–172792)
- `Amount`: Transaction amount in Euros (range: 0–2500)
- `V1–V28`: Anonymized PCA components (original features hidden)

⚠️ Use dot-decimals (e.g., 12.5) — do NOT use commas (e.g., 12,5).
"""



# Build Gradio app with two tabs (single and batch)
single_demo = gr.Interface(
    fn=predict_with_shap,
    inputs=get_inputs(),
    outputs=["text", "image"],
    title="Credit Card Fraud Detection Dashboard",
    description=description,
    article=f"""
    <hr>
    <p style='text-align:center;'>
    <em>Built by Mohamed Derouiche &mdash;
    <a href='https://github.com/mderouiche7' target='_blank'>GitHub</a> |
    <a href='https://www.linkedin.com/in/mohamed-derouiche-ba1843294' target='_blank'>LinkedIn</a> </em>
    </p>
    """
)


# Batch prediction interface
batch_demo = gr.Interface(
    fn=batch_predict,
    inputs=gr.File(label="Upload CSV with features"),
    outputs=gr.File(label="Download Predictions CSV"),
    title="Batch Fraud Prediction"
)

# Combine into a tabbed interface
tabs = gr.TabbedInterface(
    interface_list=[single_demo, batch_demo],
    tab_names=["Single Prediction", "Batch Prediction"]
)
if __name__ == "__main__":
    tabs.launch()
