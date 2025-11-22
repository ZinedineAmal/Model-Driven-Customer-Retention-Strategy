import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Dashboard + Prediction", layout="wide")
st.title("📊 Customer Churn Dashboard & Prediction")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clean_df.csv")

clean_df = load_data()

# ---------------------------------------------------------------
# COLUMN DEFINITIONS — FIXED BASED ON X_TRAIN YOU PROVIDED
# ---------------------------------------------------------------
final_cols = [
    'gender', 'age', 'senior_citizen', 'dependents', 'number_of_dependents',
    'married', 'city', 'phone_service', 'internet_service',
    'online_security', 'online_backup', 'device_protection',
    'premium_tech_support', 'streaming_tv', 'streaming_movies',
    'streaming_music', 'internet_type', 'contract', 'paperless_billing',
    'payment_method', 'monthly_charges',
    'avg_monthly_long_distance_charges', 'tenure', 'multiple_lines',
    'avg_monthly_gb_download', 'unlimited_data', 'offer',
    'referred_a_friend', 'number_of_referrals', 'satisfaction_score',
    'cltv', 'churn_score'
]

numeric_cols = [
    'age', 'number_of_dependents', 'monthly_charges',
    'avg_monthly_long_distance_charges', 'tenure',
    'avg_monthly_gb_download', 'number_of_referrals',
    'satisfaction_score', 'cltv', 'churn_score'
]

cat_cols = [col for col in final_cols if col not in numeric_cols]

# ---------------------------------------------------------------
# PREPROCESSOR (ONEHOT + SCALER)
# ---------------------------------------------------------------
preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

# Fit preprocess on clean_df → must match model training
preprocess.fit(clean_df[final_cols])

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.header("📌 Filter Options (EDA)")

clean_df['age_group'] = pd.cut(
    clean_df['age'],
    bins=[0, 27, 44, 60, float('inf')],
    labels=['Gen Z','Millennials','Generation X','Senior Citizen']
)

clean_df['tenure_group'] = pd.cut(
    clean_df['tenure'],
    bins=[0,6,12,float('inf')],
    labels=['0-6 months','6-12 months','>12 months']
)

gender_filter = st.sidebar.multiselect("Gender", clean_df['gender'].unique(), clean_df['gender'].unique())
age_group_filter = st.sidebar.multiselect("Age Group", ['Gen Z','Millennials','Generation X','Senior Citizen'], ['Gen Z','Millennials','Generation X','Senior Citizen'])
tenure_group_filter = st.sidebar.multiselect("Tenure Group", ['0-6 months','6-12 months','>12 months'], ['0-6 months','6-12 months','>12 months'])
status_filter = st.sidebar.multiselect("Churn Status", [0,1], [0,1])

filtered_df = clean_df[
    (clean_df['gender'].isin(gender_filter)) &
    (clean_df['age_group'].isin(age_group_filter)) &
    (clean_df['tenure_group'].isin(tenure_group_filter)) &
    (clean_df['churn_value'].isin(status_filter))
]

# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Churn Analysis", "Loyal Analysis", "Correlation Heatmap", "Prediction"
])

# ---------------------------------------------------------------
# TAB 1 : OVERVIEW
# ---------------------------------------------------------------
with tab1:
    st.subheader("📈 Churn Overview")

    churn_counts = filtered_df['churn_value'].value_counts().rename({0:"No Churn",1:"Churn"})

    col1, col2 = st.columns(2)
    col1.metric("Churned Customers", f"{churn_counts.get('Churn',0)}",
                f"{churn_counts.get('Churn',0)/len(filtered_df)*100:.1f}%")
    col2.metric("Stayed Customers", f"{churn_counts.get('No Churn',0)}",
                f"{churn_counts.get('No Churn',0)/len(filtered_df)*100:.1f}%")

    fig = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        color=churn_counts.index,
        color_discrete_map={'No Churn':'indigo','Churn':'salmon'},
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 Insights"):
        st.write("- Churn rate sekitar 26–30% tergantung filter.")
        st.write("- Sebagian besar pelanggan tetap loyal.")

# ---------------------------------------------------------------
# TAB 2 : CHURN ANALYSIS
# ---------------------------------------------------------------
with tab2:
    st.subheader("🚨 Churned Customer Insights")
    churned = filtered_df[filtered_df['churn_value']==1]

    gender_count = churned['gender'].value_counts().reset_index(name='count')
    fig = px.bar(gender_count, x='gender', y='count', color='gender',
                 color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)

    age_count = churned['age_group'].value_counts().reset_index(name='count')
    fig = px.bar(age_count, x='age_group', y='count', color='age_group',
                 color_discrete_sequence=px.colors.sequential.Oranges)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 Insights"):
        st.write("- Senior Citizen memiliki tingkat churn tertinggi.")
        st.write("- Pelanggan tanpa dependents lebih rentan churn.")

# ---------------------------------------------------------------
# TAB 3 : LOYAL ANALYSIS
# ---------------------------------------------------------------
with tab3:
    st.subheader("💎 Loyal Customers Insights")
    loyal = filtered_df[filtered_df['churn_value']==0]

    if 'offer' in loyal.columns:
        offer_group = (
            loyal.groupby('offer')['number_of_referrals']
            .agg(['count', lambda x: (x>=3).sum()])
            .reset_index()
        )
        offer_group.columns = ['Offer','Total Customers','With >=3 Referrals']
        st.dataframe(offer_group)

    with st.expander("💡 Insights"):
        st.write("- Banyak pelanggan loyal tidak menerima offer.")
        st.write("- Referral tinggi biasanya datang dari pelanggan yang puas.")

# ---------------------------------------------------------------
# TAB 4 : CORRELATION
# ---------------------------------------------------------------
with tab4:
    st.subheader("🔗 Correlation Matrix of Numeric Columns")
    numeric_columns = filtered_df.select_dtypes(include=['int64','float64']).columns

    if len(numeric_columns) > 0:
        corr_matrix = filtered_df[numeric_columns].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak tersedia kolom numerik.")

# ---------------------------------------------------------------
# TAB 5 : PREDICTION
# ---------------------------------------------------------------
with tab5:
    st.subheader("🤖 Predict Churn for a New Customer")

    with st.form("prediction_form"):
        inputs = {}

        for col in final_cols:
            if col in numeric_cols:
                inputs[col] = st.number_input(col, value=float(clean_df[col].median()))
            else:
                inputs[col] = st.selectbox(col, clean_df[col].unique())

        submitted = st.form_submit_button("Predict")

    if submitted:
        user_input = pd.DataFrame({k:[v] for k,v in inputs.items()})
        user_input = user_input[final_cols]

        with st.spinner("Processing..."):
            user_processed = preprocess.transform(user_input)

            with open("model_churn_XGB (1).pkl","rb") as f:
                model = pickle.load(f)
            st.write("Shape user_processed:", user_processed.shape)

            import xgboost as xgb
            st.write("Model expects features:", model.get_booster().num_features())

            prediction = model.predict(user_processed)
            prediction_proba = model.predict_proba(user_processed)

        st.write("Predicted Churn:", "Yes" if prediction[0]==1 else "No")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[0][1]*100,
            title={'text':"Probability of Churn (%)"},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':'red' if prediction[0]==1 else 'green'}}
        ))
        st.plotly_chart(fig, use_container_width=True)
