import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Interactive Churn Dashboard", layout="wide")
st.title("Interactive Customer Churn Dashboard")

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("clean_df.csv")

# ---------------- Preprocessing ----------------
# Age Group
df['age_group'] = pd.cut(df['age'], bins=[0,27,44,60,float('inf')],
                         labels=['Gen Z','Millennials','Generation X','Senior Citizen'])
# Tenure Group
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,6,12,float('inf')],
                            labels=['0-6 months','6-12 months','>12 months'])

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filter Options")
gender_filter = st.sidebar.multiselect("Select Gender", options=df['gender'].unique(), default=df['gender'].unique())
age_group_filter = st.sidebar.multiselect("Select Age Group", options=df['age_group'].dropna().unique(), default=df['age_group'].dropna().unique())
tenure_group_filter = st.sidebar.multiselect("Select Tenure Group", options=df['tenure_group'].dropna().unique(), default=df['tenure_group'].dropna().unique())
status_filter = st.sidebar.multiselect("Customer Status", options=df['churn_value'].unique(), default=df['churn_value'].unique())

# Apply filters
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
    churn_counts = filtered_df['churn_value'].value_counts()
    
    col1, col2 = st.columns(2)
    col1.metric("Churned Customers", f"{churn_counts.get(1,0)}", f"{churn_counts.get(1,0)/len(filtered_df)*100:.1f}%")
    col2.metric("Stayed / Loyal Customers", f"{churn_counts.get(0,0)}", f"{churn_counts.get(0,0)/len(filtered_df)*100:.1f}%")
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(churn_counts, labels=['No Churn','Churn'], autopct='%1.1f%%', colors=['indigo','salmon'], explode=[0,0.05])
    st.pyplot(fig)

# ---------------- Tab 2: Churn Analysis ----------------
with tab2:
    churned = filtered_df[filtered_df['churn_value']==1]
    st.subheader("Churned Customer Insights")
    
    # Gender distribution
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='gender', data=churned, palette='Reds', ax=ax)
    ax.set_title("Churned Customers by Gender")
    st.pyplot(fig)
    st.info("Insight: Gender does not appear to be a significant factor in churn; nearly evenly split.")
    
    # Age group
    fig, ax = plt.subplots(figsize=(6,4))
    churned['age_group'].value_counts().plot(kind='bar', color='orange', ax=ax)
    ax.set_title("Churned Customers by Age Group")
    st.pyplot(fig)
    st.info("Insight: Senior Citizens have the highest churn rate.")
    
    # Online services selection
    online_services = ['internet_service', 'online_security', 'online_backup', 'device_protection', 
                       'premium_tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music']
    service_select = st.multiselect("Select online services to plot", online_services, default=online_services[:3])
    
    for col in service_select:
        if col in churned.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            churned[col].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette('rocket', len(churned[col].unique())))
            ax.set_ylabel('')
            ax.set_title(f"Churn Distribution by {col}")
            st.pyplot(fig)
    
    # Top churn reasons
    if 'churn_reason' in churned.columns:
        st.subheader("Top Churn Reasons")
        top_reasons = churned['churn_reason'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=top_reasons.values, y=top_reasons.index, palette='magma', ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Reason")
        st.pyplot(fig)

# ---------------- Tab 3: Loyal Analysis ----------------
with tab3:
    loyal = filtered_df[filtered_df['churn_value']==0]
    st.subheader("Loyal Customers Insights")
    
    if 'offer' in loyal.columns:
        offer_group = loyal.groupby('offer')['number_of_referrals'].agg(['count', lambda x: (x>=3).sum()]).reset_index()
        offer_group.columns = ['Offer','Total Customers','With >=3 Referrals']
        st.dataframe(offer_group)
        st.info("Insight: Loyal customers often refer friends even without offers, showing engagement.")

# ---------------- Tab 4: Correlation Heatmap ----------------
with tab4:
    st.subheader("Correlation Matrix of Numeric Columns")
    
    numeric_columns = filtered_df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_columns) > 0:
        corr_matrix = filtered_df[numeric_columns].dropna().corr()
        fig, ax = plt.subplots(figsize=(16,12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'fontsize':8}, ax=ax)
        st.pyplot(fig)
        st.info("Insight: Helps identify correlations between numeric features.")
    else:
        st.warning("Tidak ada kolom numerik untuk correlation matrix.")
