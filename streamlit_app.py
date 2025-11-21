import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# LOAD MODEL & DATA
# ================================
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")

@st.cache_resource
def load_data():
    return pd.read_csv("df_clean.csv")

model = load_model()
df = load_data()

# detect columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# ================================
# SIDEBAR NAVIGATION
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Prediction", "EDA"])

# ================================
# PAGE 1 — PREDICTION
# ================================
if page == "Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write("Isi informasi customer untuk memprediksi apakah mereka akan churn atau tidak.")

    input_data = {}

    with st.form("form_customer"):
        st.subheader("Customer Details")

        # numeric
        for col in num_cols:
            default_val = float(df[col].median())
            input_data[col] = st.number_input(col, value=default_val)

        # categorical
        for col in cat_cols:
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("🔍 Prediction Result")

        if pred == 1:
            st.error(f"⚠️ Customer BERISIKO CHURN.\n\n**Probability: {prob:.2f}**")
        else:
            st.success(f"✅ Customer TIDAK churn.\n\n**Probability: {prob:.2f}**")

# ================================
# PAGE 2 — EDA
# ================================
else:

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write(f"Jumlah baris: **{df.shape[0]}**")
    st.write(f"Jumlah kolom: **{df.shape[1]}**")

    st.markdown("---")

    # numerical distribution
    st.subheader("🔢 Numerical Features Distribution")

    selected_num = st.selectbox("Pilih kolom numerik:", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # categorical distribution
    st.subheader("🔤 Categorical Features Distribution")

    selected_cat = st.selectbox("Pilih kolom kategorik:", cat_cols)

    fig2, ax2 = plt.subplots()
    df[selected_cat].value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")

    # correlation heatmap
    st.subheader("📈 Correlation Heatmap")

    corr = df[num_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.markdown("---")

    st.success("EDA Selesai! Silakan pindah ke halaman Prediction.")
