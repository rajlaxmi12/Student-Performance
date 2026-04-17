import os
import sys
import streamlit as st
import pandas as pd

# Add src folder to Python path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, "src")
sys.path.append(src_path)

from preprocessing import load_and_clean_data
from model_trainer import train_and_analyze


st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Analysis Dashboard")
st.write("This app analyzes student performance and predicts whether a student will pass or fail.")

# Load and process data
X_train, X_test, y_train, y_test = load_and_clean_data()

# Train model and get results
model, accuracy, report, cm, feature_df, plot_path = train_and_analyze(
    X_train, X_test, y_train, y_test
)

# Show basic information
st.subheader("Dataset Information")
st.write(f"Training samples: {X_train.shape[0]}")
st.write(f"Testing samples: {X_test.shape[0]}")
st.write(f"Number of features: {X_train.shape[1]}")

# Accuracy
st.subheader("Model Accuracy")
st.success(f"Accuracy: {accuracy:.2f}")

# Classification report
st.subheader("Classification Report")
st.text(report)

# Confusion matrix
st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(
    cm,
    index=["Actual Fail", "Actual Pass"],
    columns=["Predicted Fail", "Predicted Pass"]
)
st.dataframe(cm_df)

# Feature importance table
st.subheader("Top 10 Important Features")
st.dataframe(feature_df.reset_index(drop=True))

# Feature importance image
st.subheader("Feature Importance Chart")
st.image(plot_path, caption="Top 10 Important Features", use_container_width=True)