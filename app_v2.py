import streamlit as st
import joblib
import numpy as np
# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("fraud_model.pkl")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1 {
    color: #00E676;
    text-align: center;
}

[data-testid="stSidebar"] {
    background-color: #161B22;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
### Detect Fraudulent Credit Card Transactions using Machine Learning

This application uses a **Random Forest Classifier** to predict whether a credit card transaction is **Genuine** or **Fraudulent**.
""")

st.divider()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📊 Project Information")

    st.write("**Model:** Random Forest")
    st.write("**Dataset:** Credit Card Fraud Detection")
    st.write("**Accuracy:** 99.96%")
    st.write("**Developer:** Rashika Sharma")

    st.divider()

    st.info(
        "This application predicts whether a transaction is Genuine or Fraudulent."
    )

    st.divider()

    st.subheader("📈 Dataset Statistics")

    st.write("**Total Transactions:** 284,807")
    st.write("**Genuine Transactions:** 284,315")
    st.write("**Fraud Transactions:** 492")

    # 👇 Paste Step 10 here

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Language:** Python")
    st.write("**Framework:** Streamlit")
    st.write("**ML Library:** Scikit-learn")
# ----------------------------
# Dashboard
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📊 Accuracy", "99.96%")

with col2:
    st.metric("🤖 Model", "Random Forest")

with col3:
    st.metric("💳 Fraud Cases", "492")

st.divider()
# ----------------------------
# Transaction Input Form
# ----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:

    st.subheader("📝 Transaction Details")

    time = st.number_input(
        "Transaction Time",
        value=0.0,
        key="time"
    )

    amount = st.number_input(
        "Transaction Amount",
        value=0.0,
        key="amount"
    )
with st.expander("⚙️ Advanced Features (V1 - V28)"):

    features = []

    col_a, col_b = st.columns(2)

    for i in range(1, 15):
        with col_a:
            value = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"v{i}"
            )
            features.append(value)

    for i in range(15, 29):
        with col_b:
            value = st.number_input(
                f"V{i}",
                value=0.0,
                key=f"v{i}"
            )
            features.append(value)
with right_col:

    st.subheader("ℹ️ Instructions")

    st.info("""
1. Enter the transaction details.

2. Fill the PCA features.

3. Click **Predict Transaction**.
""")
    # ----------------------------
# Prediction
# ----------------------------
st.divider()

if st.button("🔍 Predict Transaction", use_container_width=True):

    input_data = np.array([[
        time,
        *features,
        amount
    ]])

    prediction = model.predict(input_data)

    # Check if the model supports probability prediction
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)
        confidence = np.max(probability) * 100
    else:
        confidence = None

    st.divider()
    st.subheader("📋 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction[0] == 0:
            st.success("✅ Genuine Transaction")
        else:
            st.error("🚨 Fraudulent Transaction Detected")

    with result_col2:
        if confidence is not None:
            st.metric(
                label="🎯 Confidence",
                value=f"{confidence:.2f}%"
            )
            st.divider()

with st.expander("ℹ️ About this Project"):

    st.markdown("""
### 📌 Project Overview

This project detects fraudulent credit card transactions using a Machine Learning model trained on the Credit Card Fraud Detection dataset.

### ✨ Features

- Random Forest Classifier
- Real-time Prediction
- Confidence Score
- Interactive Streamlit Dashboard
- Responsive User Interface

### 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Joblib
""")
        # ----------------------------
# Footer
# ----------------------------
st.divider()

st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:16px;'>
        💳 Credit Card Fraud Detection System <br>
        Developed by <b>Rashika Sharma</b>
    </div>
    """,
    unsafe_allow_html=True
)