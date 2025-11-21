import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Customer Churn Dashboard + Prediction", layout="wide")
st.title("Interactive Customer Churn Dashboard & Prediction")

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    return pd.read_csv("clean_df.csv")  # ganti dengan GitHub raw CSV link

clean_df = load_data()

# ---------------- Columns ----------------
# Kolom final yang dipakai model XGBoost
final_cols = ['age','number_of_dependents','monthly_charges','avg_monthly_long_distance_charges',
              'tenure','avg_monthly_gb_download','number_of_referrals','satisfaction_score',
              'cltv','churn_score','gender','senior_citizen','dependents','city','phone_service',
              'online_security','online_backup','device_protection','premium_tech_support',
              'streaming_tv','streaming_movies','streaming_music','internet_type','contract',
              'paperless_billing','payment_method','multiple_lines','unlimited_data','offer']

numeric_cols = ['age','number_of_dependents','monthly_charges','avg_monthly_long_distance_charges',
                'tenure','avg_monthly_gb_download','number_of_referrals','satisfaction_score',
                'cltv','churn_score']

cat_cols = [col for col in final_cols if col not in numeric_cols]

# ColumnTransformer fit ke clean_df
preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
])
preprocess.fit(clean_df[final_cols])

# ---------------- Sidebar Filters for EDA ----------------
st.sidebar.header("Filter Options (EDA)")
gender_filter = st.sidebar.multiselect("Gender", options=clean_df['gender'].unique(), default=clean_df['gender'].unique())
age_group_filter = st.sidebar.multiselect("Age Group", options=['Gen Z','Millennials','Generation X','Senior Citizen'],
                                          default=['Gen Z','Millennials','Generation X','Senior Citizen'])
tenure_group_filter = st.sidebar.multiselect("Tenure Group", options=['0-6 months','6-12 months','>12 months'],
                                             default=['0-6 months','6-12 months','>12 months'])
status_filter = st.sidebar.multiselect("Churn Status", options=[0,1], default=[0,1])

# Add age_group & tenure_group for filtering
clean_df['age_group'] = pd.cut(clean_df['age'], bins=[0,27,44,60,float('inf')],
                              labels=['Gen Z','Millennials','Generation X','Senior Citizen'])
clean_df['tenure_group'] = pd.cut(clean_df['tenure'], bins=[0,6,12,float('inf')],
                                 labels=['0-6 months','6-12 months','>12 months'])

filtered_df = clean_df[
    (clean_df['gender'].isin(gender_filter)) &
    (clean_df['age_group'].isin(age_group_filter)) &
    (clean_df['tenure_group'].isin(tenure_group_filter)) &
    (clean_df['churn_value'].isin(status_filter))
]

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Churn Analysis", "Loyal Analysis", "Correlation Heatmap", "Prediction"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    st.subheader("Churn Overview")
    churn_counts = filtered_df['churn_value'].value_counts().rename({0:"No Churn",1:"Churn"})
    st.metric("Churned Customers", f"{churn_counts.get('Churn',0)}", f"{churn_counts.get('Churn',0)/len(filtered_df)*100:.1f}%")
    st.metric("Stayed Customers", f"{churn_counts.get('No Churn',0)}", f"{churn_counts.get('No Churn',0)/len(filtered_df)*100:.1f}%")
    fig = px.pie(values=churn_counts.values, names=churn_counts.index, color=churn_counts.index,
                 color_discrete_map={'No Churn':'indigo','Churn':'salmon'}, hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 2: Churn Analysis ----------------
with tab2:
    churned = filtered_df[filtered_df['churn_value']==1]
    st.subheader("Churned Customer Insights")
    
    # Gender
    gender_count = churned['gender'].value_counts().rename_axis('gender').reset_index(name='count')
    fig = px.bar(gender_count, x='gender', y='count', color='gender', 
                 color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)
    
    # Age group
    age_count = churned['age_group'].value_counts().rename_axis('age_group').reset_index(name='count')
    fig = px.bar(age_count, x='age_group', y='count', color='age_group', 
                 color_discrete_sequence=px.colors.sequential.Oranges)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 3: Loyal Analysis ----------------
with tab3:
    loyal = filtered_df[filtered_df['churn_value']==0]
    st.subheader("Loyal Customers Insights")
    if 'offer' in loyal.columns:
        offer_group = loyal.groupby('offer')['number_of_referrals'].agg(['count', lambda x: (x>=3).sum()]).reset_index()
        offer_group.columns = ['Offer','Total Customers','With >=3 Referrals']
        st.dataframe(offer_group)

# ---------------- Tab 4: Correlation Heatmap ----------------
with tab4:
    st.subheader("Correlation Matrix of Numeric Columns")
    numeric_columns = filtered_df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_columns) > 0:
        corr_matrix = filtered_df[numeric_columns].dropna().corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada kolom numerik untuk correlation matrix")

# ---------------- Tab 5: Prediction ----------------
with tab5:
    st.subheader("Predict Churn for a New Customer")
    
    # Input form
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
        
        # Pastikan kolom urut sesuai model
        user_input = user_input[final_cols]
        
        # Preprocess
        user_processed = preprocess.transform(user_input)
        
        # Load model
        with open("model_churn.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Predict
        prediction = model.predict(user_processed)
        prediction_proba = model.predict_proba(user_processed)
        
        st.write("Predicted Churn:", "Yes" if prediction[0]==1 else "No")
        st.write("Probability of Churn:", f"{prediction_proba[0][1]*100:.2f}%")
