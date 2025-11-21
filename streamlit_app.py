import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# LOAD ASSETS
# ================================
@st.cache_resource
def load_preprocess():
    return joblib.load("preprocess.pkl")

@st.cache_resource
def load_model():
    return joblib.load("model_churn.pkl")

@st.cache_resource
def load_data():
    return pd.read_csv("df_clean.csv")

preprocess = load_preprocess()
model = load_model()
df = load_data()

# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Prediction", "EDA"])

# ================================
# PAGE 1 — PREDICTION
# ================================
if page == "Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write("Isi informasi customer untuk memprediksi apakah mereka akan churn atau tidak.")

    # detect columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # input form
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

    # prediction
    if submitted:
        input_df = pd.DataFrame([input_data])
        X_trans = preprocess.transform(input_df)
        pred = model.predict(X_trans)[0]
        prob = model.predict_proba(X_trans)[0][1]

        st.subheader("🔍 Prediction Result")

        if pred == 1:
            st.error(f"⚠️ Customer BERISIKO CHURN.\n\n**Probability: {prob:.2f}**")
        else:
            st.success(f"✅ Customer TIDAK churn.\n\n**Probability: {prob:.2f}**")


# ================================
# PAGE 2 — EDA
# ================================
elif page == "EDA":

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write(f"Jumlah baris: **{df.shape[0]}**")
    st.write(f"Jumlah kolom: **{df.shape[1]}**")

    st.markdown("---")

    # =====================
    # DISTRIBUTION NUMERIC
    # =====================
    st.subheader("🔢 Numerical Features Distribution")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    selected_num = st.selectbox("Pilih kolom numerik:", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # =====================
    # DISTRIBUTION CATEGORICAL
    # =====================
    st.subheader("🔤 Categorical Features Distribution")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    selected_cat = st.selectbox("Pilih kolom kategorik:", cat_cols)

    fig2, ax2 = plt.subplots()
    df[selected_cat].value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    st.markdown("---")

    # =====================
    # CORRELATION HEATMAP
    # =====================
    st.subheader("📈 Correlation Heatmap")

    corr = df[num_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.markdown("---")

    st.success("EDA Selesai! Kamu bisa pindah ke halaman Prediction di sidebar.")

