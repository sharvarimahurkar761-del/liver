# main.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_models

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Livera",
    layout="centered",
    page_icon="🩺"
)

# ------------------ Custom Styles ------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        font-family: 'Arial', sans-serif;
        color: #1f2937;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-size: 18px;
    }
    .stSidebar .sidebar-content {
        background-color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Title ------------------
st.title("🩺 Livera")
st.markdown("#### Predict potential liver disease risk with interactive AI")

# ------------------ Load Models ------------------
rf_model, gb_model = train_models()

# ------------------ Sidebar Input ------------------
st.sidebar.header("Enter Patient Data")
def user_input():
    Age = st.sidebar.slider("Age", 20, 80, 40)
    Bilirubin = st.sidebar.slider("Bilirubin", 0.3, 3.0, 1.0)
    AlkPhos = st.sidebar.slider("Alkaline Phosphatase", 150, 400, 200)
    SGPT = st.sidebar.slider("SGPT (ALT)", 10, 100, 30)
    Albumin = st.sidebar.slider("Albumin", 2.5, 5.0, 4.0)
    A_G_Ratio = st.sidebar.slider("Albumin/Globulin Ratio", 0.8, 2.5, 1.2)
    Total_Proteins = st.sidebar.slider("Total Proteins", 5.0, 8.0, 6.5)
    Age_Bilirubin_Ratio = Age / Bilirubin
    return pd.DataFrame({
        "Age":[Age],
        "Bilirubin":[Bilirubin],
        "Alkaline_Phosphotase":[AlkPhos],
        "SGPT_Alanine_Aminotransferase":[SGPT],
        "Albumin":[Albumin],
        "Albumin_and_Globulin_Ratio":[A_G_Ratio],
        "Total_Proteins":[Total_Proteins],
        "Age_Bilirubin_Ratio":[Age_Bilirubin_Ratio]
    })

input_df = user_input()

# ------------------ Predictions ------------------
rf_pred = rf_model.predict(input_df)[0]
gb_pred = gb_model.predict(input_df)[0]
rf_prob = rf_model.predict_proba(input_df)[0][1] * 100
gb_prob = gb_model.predict_proba(input_df)[0][1] * 100

# ------------------ Display Predictions ------------------
st.subheader("Prediction")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"<div style='padding:15px; background-color: {'#e74c3c' if rf_pred==1 else '#2ecc71'}; color:white; font-size:20px; text-align:center;'>"
        f"Random Forest:<br>{'At Risk' if rf_pred==1 else 'Healthy'}<br>{rf_prob:.1f}% probability</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div style='padding:15px; background-color: {'#e74c3c' if gb_pred==1 else '#2ecc71'}; color:white; font-size:20px; text-align:center;'>"
        f"Gradient Boosting:<br>{'At Risk' if gb_pred==1 else 'Healthy'}<br>{gb_prob:.1f}% probability</div>",
        unsafe_allow_html=True
    )

# ------------------ Feature Importance ------------------
st.subheader("Feature Importance (Random Forest)")
feat_imp = pd.Series(rf_model.feature_importances_, index=input_df.columns)
fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax, palette="viridis")
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.tick_params(labelsize=10)
st.pyplot(fig)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<p style='font-size:16px;'>💡 This interactive app demonstrates how <b>machine learning can turn complex health data into actionable insights</b> in a clean, user-friendly interface.</p>",
    unsafe_allow_html=True
)  