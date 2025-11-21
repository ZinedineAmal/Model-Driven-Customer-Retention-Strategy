import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Interactive Churn Dashboard", layout="wide")
st.title("Interactive Customer Churn Dashboard")

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("data.csv")

# ---------------- Preprocessing ----------------
df['age_group'] = pd.cut(df['age'], bins=[0,27,44,60,float('inf')],
                         labels=['Gen Z','Millennials','Generation X','Senior Citizen'])
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,6,12,float('inf')],
                            labels=['0-6 months','6-12 months','>12 months'])

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filter Options")
gender_filter = st.sidebar.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
age_group_filter = st.sidebar.multiselect("Age Group", options=df['age_group'].dropna().unique(), default=df['age_group'].dropna().unique())
tenure_group_filter = st.sidebar.multiselect("Tenure Group", options=df['tenure_group'].dropna().unique(), default=df['tenure_group'].dropna().unique())
status_filter = st.sidebar.multiselect("Churn Status", options=df['churn_value'].unique(), default=df['churn_value'].unique())

filtered_df = df[
    (df['gender'].isin(gender_filter)) &
    (df['age_group'].isin(age_group_filter)) &
    (df['tenure_group'].isin(tenure_group_filter)) &
    (df['churn_value'].isin(status_filter))
]

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Churn Analysis", "Loyal Analysis", "Correlation Heatmap"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    st.subheader("Churn Overview")
    churn_counts = filtered_df['churn_value'].value_counts().rename({0:"No Churn",1:"Churn"})
    st.metric("Churned Customers", f"{churn_counts.get('Churn',0)}", f"{churn_counts.get('Churn',0)/len(filtered_df)*100:.1f}%")
    st.metric("Stayed Customers", f"{churn_counts.get('No Churn',0)}", f"{churn_counts.get('No Churn',0)/len(filtered_df)*100:.1f}%")
    
    # Pie Chart Plotly
    fig = px.pie(values=churn_counts.values, names=churn_counts.index, color=churn_counts.index,
                 color_discrete_map={'No Churn':'indigo','Churn':'salmon'}, hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 2: Churn Analysis ----------------
with tab2:
    churned = filtered_df[filtered_df['churn_value']==1]
    st.subheader("Churned Customer Insights")
    
    # Gender
    gender_count = churned['gender'].value_counts().rename_axis('gender').reset_index(name='count')
    fig = px.bar(gender_count, x='gender', y='count', color='gender', color_discrete_sequence=px.colors.sequential.Reds)
    st.plotly_chart(fig, use_container_width=True)
    
    # Age group
    age_count = churned['age_group'].value_counts().rename_axis('age_group').reset_index(name='count')
    fig = px.bar(age_count, x='age_group', y='count', color='age_group', color_discrete_sequence=px.colors.sequential.Orange)
    st.plotly_chart(fig, use_container_width=True)
    
    # Online Services Pie Charts
    online_services = ['internet_service', 'online_security', 'online_backup', 'device_protection', 
                       'premium_tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music']
    service_select = st.multiselect("Select online services", online_services, default=online_services[:3])
    
    for col in service_select:
        if col in churned.columns:
            data = churned[col].value_counts().rename_axis(col).reset_index(name='count')
            fig = px.pie(data, names=col, values='count', hole=0.3, color=col, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
    
    # Top Churn Reasons
    if 'churn_reason' in churned.columns:
        top_reasons = churned['churn_reason'].value_counts().head(10).rename_axis('reason').reset_index(name='count')
        fig = px.bar(top_reasons, x='count', y='reason', orientation='h', color='count', color_continuous_scale='magma')
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
