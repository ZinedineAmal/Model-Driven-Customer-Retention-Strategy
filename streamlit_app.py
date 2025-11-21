import streamlit as st
import pandas as pd
import joblib

# ================================
# FILE PATH
# ================================
MODEL_PATH = "model_churn.pkl"
PREPROCESS_PATH = "preprocess.pkl"

# ================================
# LOAD ASSETS
# ================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocess():
    return joblib.load(PREPROCESS_PATH)

model = load_model()
preprocess = load_preprocess()

# Ambil feature names INPUT ke preprocess (bukan ke model)
original_features = preprocess.feature_names_in_

# ================================
# INPUT FORM
# ================================
st.title("Customer Churn Prediction (Final Fixed Version)")

input_data = {}

with st.form("form_input"):
    st.write("Isi data customer sesuai fitur:")

    for col in original_features:
        # numeric → number_input
        input_data[col] = st.text_input(col)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert to DF
    input_df = pd.DataFrame([input_data])

    # =========================
    # CONVERT AUTO TYPES
    # =========================
    for col in input_df.columns:
        try:
            # Coba convert ke numeric
            input_df[col] = pd.to_numeric(input_df[col])
        except:
            # Kalau gagal → biarkan jadi string (categorical)
            pass

    st.subheader("Raw Input")
    st.write(input_df)

    # =========================
    # APPLY PREPROCESS
    # =========================
    processed = preprocess.transform(input_df)

    st.subheader("Processed Input (Encoded)")
    st.write(processed)

    # =========================
    # PREDICTION
    # =========================
    pred = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ Customer BERISIKO churn. Probability = {proba:.2f}")
    else:
        st.success(f"✅ Customer aman. Probability = {proba:.2f}")
