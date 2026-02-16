# main.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import generate_liver_dataset
from model import train_models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# -------------------------------
# Streamlit page config & custom CSS
# -------------------------------
st.set_page_config(page_title="Liver Disease Detection", layout="wide")

st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #f0f8ff;  /* light blue */
        font-family: 'Segoe UI', sans-serif;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0b3d91;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #e6f2ff;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #0b3d91;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title
# -------------------------------
st.markdown("## 🩺 Liver Disease Detection")

# -------------------------------
# Load dataset
# -------------------------------
df = generate_liver_dataset()
X = df.drop('target', axis=1)
y = df['target']

# -------------------------------
# Train models
# -------------------------------
rf_model, gb_model, X_train, X_test, y_train, y_test, _, _ = train_models(X, y)

# -------------------------------
# Model comparison
# -------------------------------
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
rf_auc = auc(*roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])[:2])
gb_auc = auc(*roc_curve(y_test, gb_model.predict_proba(X_test)[:,1])[:2])

st.markdown("### Model Comparison")
st.write(f"**Random Forest → Accuracy:** {rf_acc:.2f}, **AUC:** {rf_auc:.2f}")
st.write(f"**Gradient Boosting → Accuracy:** {gb_acc:.2f}, **AUC:** {gb_auc:.2f}")

# -------------------------------
# Evaluation Metrics – Random Forest
# -------------------------------
st.markdown("### 📊 Evaluation Metrics – Random Forest")
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

# Side-by-side plots
col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ROC Curve
with col2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1],[0,1],'r--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)

# Feature Importance
st.markdown("### 📝 Feature Importance")
feat_imp = rf_model.feature_importances_
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.barplot(x=feat_imp, y=X_test.columns, ax=ax3)
ax3.set_title("Random Forest Feature Importance")
st.pyplot(fig3)

# -------------------------------
# Interactive Prediction
# -------------------------------
st.markdown("### 🖥 Predict Your Own Sample")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

model_choice = st.radio("Select Model for Prediction", ["Random Forest", "Gradient Boosting"])
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    selected_model = rf_model if model_choice=="Random Forest" else gb_model
    pred = selected_model.predict(input_df)[0]
    proba = selected_model.predict_proba(input_df)[0][pred]
    label = "Liver Disease" if pred==1 else "Healthy"
    st.success(f"Prediction using {model_choice}: **{label}** (Confidence: {proba:.2f})")
