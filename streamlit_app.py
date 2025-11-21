import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.write("Working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())
# ================================
# FILE PATH
# ================================
MODEL_PATH = "model_churn.pkl"
PREPROCESS_PATH = "preprocess.pkl"
DATA_PATH = "clean_df_1.csv"
FINAL_FEATURE_PATH = "final_features_FIXED.pkl"

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
st.write("ISI FINAL FEATURES YANG SEBENARNYA DILOAD:")
st.write(final_features)

# DEBUGGING — CEK WORKING DIRECTORY & FILE YANG DIPAKAI
st.write("Working directory:", os.getcwd())
st.write("FINAL_FEATURE_PATH:", FINAL_FEATURE_PATH)
st.write("File exists?", os.path.exists(FINAL_FEATURE_PATH))

try:
    f = joblib.load(FINAL_FEATURE_PATH)
    st.write("FINAL_FEATURES yang sedang dipakai Streamlit:")
    st.write(f)
except Exception as e:
    st.error(f"Gagal load FINAL_FEATURE_PATH: {e}")

st.write("CHECK FINAL FEATURES YANG LOADED:")
st.write(final_features)
st.write("Jumlah kolom:", len(final_features))

# =====================================================
# ** FIX: Ambil kolom dari preprocess, bukan dari df **
# =====================================================
num_cols = preprocess.transformers_[0][2]    # numeric columns saat training
cat_cols = preprocess.transformers_[1][2]    # categorical columns saat training

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

        # NUMERIC INPUT
        for col in num_cols:
            default_val = float(df[col].median()) if col in df.columns else 0
            input_data[col] = st.number_input(col, value=default_val)

        # CATEGORICAL INPUT
        for col in cat_cols:
            if col in df.columns:
                options = sorted(df[col].dropna().unique().tolist())
                input_data[col] = st.selectbox(col, options)
            else:
                input_data[col] = st.text_input(col)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # DATAFRAME INPUT USER
        input_df = pd.DataFrame([input_data])

        # PREPROCESS TRANSFORM
        processed = preprocess.transform(input_df)

        # KONVERSI KE DATAFRAME DENGAN NAMA KOLOM ASLI
        processed_df = pd.DataFrame(processed, columns=preprocess.get_feature_names_out())

        # DEBUGGING
        st.subheader("Debug Info")
        st.write("Raw Input DF:")
        st.write(input_df)

        st.write("Processed DF:")
        st.write(processed_df)

        st.write("Processed DF Columns:")
        st.write(processed_df.columns.tolist())

        unique_counts = processed_df.nunique()
        st.write("Unique values per column:")
        st.write(unique_counts)

        # FILTER FINAL FEATURES
        if all(col in processed_df.columns for col in final_features):
            processed_df = processed_df[final_features]
        else:
            missing = [c for c in final_features if c not in processed_df.columns]
            st.error(f"Missing columns in processed DF: {missing}")
            st.stop()

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
