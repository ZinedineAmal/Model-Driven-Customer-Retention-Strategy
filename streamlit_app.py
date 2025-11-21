import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# FILE PATH (sesuaikan kalau perlu)
# ================================
MODEL_PATH = "model_churn.pkl"
PREPROCESS_PATH = "preprocess.pkl"   # optional (app bisa jalan tanpa ini)
DATA_PATH = "clean_df_1.csv"

# ================================
# HELPERS: load
# ================================
@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

@st.cache_resource
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def load_preprocess_if_exists(path=PREPROCESS_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Gagal load preprocess.pkl: {e}")
            return None
    return None

# ================================
# LOAD ASSETS
# ================================
if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan: {MODEL_PATH}")
    st.stop()
if not os.path.exists(DATA_PATH):
    st.error(f"File data tidak ditemukan: {DATA_PATH}")
    st.stop()

model = load_model()
df = load_data()
preprocess = load_preprocess_if_exists()

# Ambil feature names dari model
try:
    model_features = model.get_booster().feature_names
except Exception:
    st.error("Gagal membaca feature names dari model. Pastikan model adalah estimator XGBoost yang valid.")
    st.stop()

# Jika preprocess ada, ambil feature names setelah transform (jika bisa)
preprocess_feature_names = None
if preprocess is not None:
    try:
        preprocess_feature_names = list(preprocess.get_feature_names_out())
    except Exception:
        preprocess_feature_names = None

# Debug kecil (bisa dihapus nanti)
st.write("Working directory:", os.getcwd())
st.write("Model feature count:", len(model_features))
st.write("Model sample features:", model_features[:8])
if preprocess_feature_names is not None:
    st.write("Preprocess feature count:", len(preprocess_feature_names))
    st.write("Preprocess sample features:", preprocess_feature_names[:8])
else:
    st.write("No usable preprocess found (preprocess_feature_names is None).")

# ================================
# SIDEBAR
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to:", ["Prediction", "EDA"])

# ================================
# FUNCTION: validate numeric frame for XGBoost
# ================================
def ensure_numeric_df_for_xgb(df_input, required_cols):
    """
    Convert columns to numeric where possible. If after conversion
    any required column still non-numeric (object) or contains NaN,
    return (False, message). Otherwise return (True, df_converted).
    """
    df_conv = df_input.copy()
    problems = []
    for c in required_cols:
        if c not in df_conv.columns:
            problems.append(f"missing column: {c}")
            continue
        if pd.api.types.is_numeric_dtype(df_conv[c]):
            continue
        # try convert
        coerced = pd.to_numeric(df_conv[c], errors="coerce")
        # if everything becomes NaN, that's bad
        if coerced.isna().all():
            problems.append(f"column not numeric and cannot convert: {c}")
        else:
            df_conv[c] = coerced
            # if still has NaN where original didn't, warn
            # (we'll handle existence of NaN later)
    # check NaNs
    nan_cols = [c for c in required_cols if df_conv[c].isna().any()]
    if problems or nan_cols:
        msg_lines = []
        if problems:
            msg_lines.append("Problems found:\n" + "; ".join(problems))
        if nan_cols:
            msg_lines.append("Columns with NaN after conversion: " + ", ".join(nan_cols))
        return False, "\n".join(msg_lines)
    return True, df_conv[required_cols]

# ================================
# PAGE — PREDICTION
# ================================
if page == "Prediction":
    st.title("Customer Churn Prediction")
    st.write("Isi informasi customer sesuai fitur model. Ikuti urutan & tipe yang diminta.")

    # Build input form based on the source of truth:
    # Prefer preprocess_feature_names (if it exists and matches model),
    # otherwise use model_features.
    use_preprocess = False
    if preprocess_feature_names is not None:
        # if names equal sets or exact list equals, prefer using preprocess
        if set(preprocess_feature_names) == set(model_features) or preprocess_feature_names == list(model_features):
            use_preprocess = True
        else:
            # if preprocess exists but feature names do not match model,
            # we will still allow using preprocess if user wants, but default to model_features
            st.info("preprocess ditemukan tapi feature names-nya berbeda dengan model. App akan pakai model feature names.")
            use_preprocess = False

    source_features = preprocess_feature_names if use_preprocess else list(model_features)

    input_data = {}
    with st.form("form_customer"):
        st.subheader("Customer Details")

        # For each expected feature, provide input widget.
        for col in source_features:
            # If column is in original df, infer widget type
            if col in df.columns:
                if df[col].dtype == "object":
                    options = sorted(df[col].dropna().unique().tolist())
                    if len(options) <= 50:
                        input_data[col] = st.selectbox(col, options)
                    else:
                        input_data[col] = st.text_input(col)
                else:
                    default_val = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0
                    input_data[col] = st.number_input(col, value=default_val)
            else:
                # fallback: free text input (user must enter numeric for numeric fields)
                input_data[col] = st.text_input(col)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])

        # If using preprocess path: transform first, then ensure columns align to model
        if use_preprocess:
            try:
                arr = preprocess.transform(input_df)
            except Exception as e:
                st.error(f"Gagal menjalankan preprocess.transform(): {e}")
                st.stop()

            # create dataframe from array with preprocess feature names
            processed_df = pd.DataFrame(arr, columns=preprocess_feature_names)

            # Try to reorder to model_features (if different order)
            missing_for_model = [c for c in model_features if c not in processed_df.columns]
            if missing_for_model:
                st.error(f"Setelah preprocess, kolom ini hilang (dibutuhkan model): {missing_for_model}")
                st.stop()

            processed_df = processed_df[model_features]  # reorder
            ok, result = ensure_numeric_df_for_xgb(processed_df, model_features)
            if not ok:
                st.error("Tipe data tidak sesuai untuk model XGBoost:\n" + result)
                st.stop()
            xgb_ready_df = result

        else:
            # No preprocess: assume model expects raw columns present in input_df
            missing_cols = [c for c in model_features if c not in input_df.columns]
            if missing_cols:
                st.error(f"Input tidak memiliki kolom yang dibutuhkan model: {missing_cols}")
                st.stop()
            # reorder
            raw_ordered = input_df[model_features]
            ok, result = ensure_numeric_df_for_xgb(raw_ordered, model_features)
            if not ok:
                st.error(
                    "Model dilatih pada data numerik tetapi input mengandung nilai non-numerik.\n"
                    "Saran:\n"
                    "- Berikan CSV input yang sesuai, atau\n"
                    "- Sediakan preprocess.pkl yang sama seperti saat training, atau\n"
                    "- Retrain model agar sinkron dengan pipeline preprocessing.\n\n"
                    + result
                )
                st.stop()
            xgb_ready_df = result

        # show debug
        st.subheader("Data yang akan dipakai model (preview)")
        st.write(xgb_ready_df.head())

        # Prediction
        try:
            pred = model.predict(xgb_ready_df)[0]
            prob = model.predict_proba(xgb_ready_df)[0][1]
        except Exception as e:
            st.error(f"Gagal prediksi model: {e}")
            st.stop()

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
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
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
