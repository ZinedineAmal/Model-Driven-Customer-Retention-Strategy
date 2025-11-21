import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# FILE PATH
# ================================
MODEL_PATH = "model_churn.pkl"
DATA_PATH = "clean_df_1.csv"

# ================================
# LOAD ASSETS
# ================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# Ambil feature names dari model (INI YANG BENAR)
model_features = model.get_booster().feature_names

# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Prediction", "EDA"])

# ================================
# PAGE — PREDICTION
# ================================
if page == "Prediction":

    st.title("Customer Churn Prediction")
    st.write("Isi informasi customer sesuai fitur model.")

    input_data = {}

    with st.form("form_customer"):
        for col in model_features:
            if col in df.columns:
                # Auto detect categorical or numeric
                if df[col].dtype == "object":
                    options = sorted(df[col].dropna().unique().tolist())
                    input_data[col] = st.selectbox(col, options)
                else:
                    default_val = float(df[col].median())
                    input_data[col] = st.number_input(col, value=default_val)
            else:
                # Fallback
                input_data[col] = st.text_input(col)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])

        # Pastikan urutan kolom sesuai model
        input_df = input_df[model_features]

        st.subheader("Processed Input")
        st.write(input_df)

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"Customer BERISIKO churn. Probability: {prob:.2f}")
        else:
            st.success(f"Customer TIDAK churn. Probability: {prob:.2f}")

# ================================
# PAGE — EDA
# ================================
else:
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")

    st.markdown("---")

    # Numerical features only
    num_cols = df.select_dtypes(include=['int64','float64']).columns

    st.subheader("Numerical Feature Distribution")
    selected_num = st.selectbox("Pilih kolom numerik:", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # Categorical features
    cat_cols = df.select_dtypes(include=['object']).columns

    st.subheader("Categorical Feature Distribution")
    selected_cat = st.selectbox("Pilih kolom kategorik:", cat_cols)

    fig2, ax2 = plt.subplots()
    df[selected_cat].value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")

    # Correlation
    st.subheader("Correlation Heatmap")
    corr = df[num_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.success("EDA selesai.")
