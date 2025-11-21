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
PREPROCESS_PATH = "preprocess.pkl"
DATA_PATH = "clean_df.csv"
FINAL_FEATURE_PATH = "final_features.pkl"

# ================================
# VALIDASI FILE
# ================================
for f in [MODEL_PATH, PREPROCESS_PATH, DATA_PATH, FINAL_FEATURE_PATH]:
    if not os.path.exists(f):
        st.error(f"File tidak ditemukan: {f}")
        st.stop()

# ================================
# LOAD ASSETS
# ================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocess():
    return joblib.load(PREPROCESS_PATH)

@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_final_features():
    return joblib.load(FINAL_FEATURE_PATH)

model = load_model()
preprocess = load_preprocess()
df = load_data()
final_features = load_final_features()

missing = [col for col in final_features if col not in processed_df.columns]
st.write("Missing columns:", missing)
st.write("Processed DF columns:", processed_df.columns.tolist())

# detect columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Prediction", "EDA"])

# ================================
# PAGE 1 — PREDICTION
# ================================
if page == "Prediction":

    st.title("Customer Churn Prediction")
    st.write("Isi informasi customer untuk prediksi churn.")

    input_data = {}

    with st.form("form_customer"):
        st.subheader("Customer Details")

        for col in num_cols:
            default_val = float(df[col].median())
            input_data[col] = st.number_input(col, value=default_val)

        for col in cat_cols:
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # DATAFRAME INPUT USER
        input_df = pd.DataFrame([input_data])

        # PREPROCESS TRANSFORM
        processed = preprocess.transform(input_df)

        # KONVERSI KE DATAFRAME DGN FINAL FEATURE
        processed_df = pd.DataFrame(processed, columns=preprocess.get_feature_names_out())

        # FILTER HANYA FINAL FEATURES DARI VIF
        processed_df = processed_df[final_features]

        # PREDIKSI
        pred = model.predict(processed_df)[0]
        prob = model.predict_proba(processed_df)[0][1]

        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"Customer BERISIKO churn. Probability: {prob:.2f}")
        else:
            st.success(f"Customer TIDAK churn. Probability: {prob:.2f}")

# ================================
# PAGE 2 — EDA
# ================================
else:
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")

    st.markdown("---")

    # numerical dist
    st.subheader("Numerical Feature Distribution")
    selected_num = st.selectbox("Pilih kolom numerik:", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # categorical dist
    st.subheader("Categorical Feature Distribution")
    selected_cat = st.selectbox("Pilih kolom kategorik:", cat_cols)

    fig2, ax2 = plt.subplots()
    df[selected_cat].value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")

    # correlation
    st.subheader("Correlation Heatmap")
    corr = df[num_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.success("EDA selesai.")
