import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("clean_df.csv")

st.title("EDA Customer Churn Analysis")

# ---------------- Heatmap Correlation ----------------
st.subheader("Correlation Matrix of Numeric Columns")
numeric_columns = df.select_dtypes(exclude=['object']).columns
corr_matrix = df[numeric_columns].corr()

fig, ax = plt.subplots(figsize=(16,12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={'fontsize':8}, ax=ax)
st.pyplot(fig)

# ---------------- Churn Pie Chart ----------------
st.subheader("Churn Rate")
churn_counts = df['churn_value'].value_counts()
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.pie(
    churn_counts,
    labels=['No Churn','Churn'],
    autopct='%1.1f%%',
    colors=['indigo','salmon'],
    explode=[0,0.05],
    textprops={"fontsize":15}
)
ax2.legend(labels=['0 = No Churn', '1 = Churn'])
st.pyplot(fig2)

st.write(f"Total Churned: {churn_counts.get(1,0)} ({churn_counts.get(1,0)/len(df)*100:.1f}%)")
st.write(f"Stayed / No Churn: {churn_counts.get(0,0)} ({churn_counts.get(0,0)/len(df)*100:.1f}%)")

# ---------------- Barplot: Gender vs Churn ----------------
st.subheader("Distribution of Churn by Gender")
fig3, ax3 = plt.subplots(figsize=(10,4))
sns.countplot(x='gender', hue='churn_value', data=df, palette='Reds', ax=ax3)
ax3.set_xlabel('Gender')
ax3.set_ylabel('Count')
st.pyplot(fig3)

# ---------------- Churned Customer Analysis ----------------
churned = df[df['churn_value']==1]
st.subheader("Churned Customer Analysis")

st.write(f"Total churned customer: {len(churned)} ({len(churned)/len(df)*100:.1f}%) from total data")

# Churned by Gender
fig4, ax4 = plt.subplots(figsize=(10,4))
sns.countplot(x='gender', data=churned, palette='Reds', ax=ax4)
ax4.set_xlabel('Gender')
ax4.set_ylabel('Count')
st.pyplot(fig4)

# Age Group
churned['age_group'] = pd.cut(churned['age'], bins=[0,27,44,60,float('inf')],
                               labels=['Gen Z','Millennials','Generation X','Senior Citizen'])
st.write("Churned by Age Group:")
st.dataframe(churned['age_group'].value_counts())

# ---------------- Online Services Pie Charts ----------------
st.subheader("Online Services among Churned Customers")
online_services = ['internet_service', 'online_security', 'online_backup', 'device_protection', 
                   'premium_tech_support', 'streaming_tv', 'streaming_movies', 'streaming_music']

fig5, axes = plt.subplots(4, 2, figsize=(16,25))
axes = axes.flatten()

for i, col in enumerate(online_services):
    churn_per_cat = churned[col].value_counts(normalize=True)
    axes[i].pie(churn_per_cat, labels=churn_per_cat.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette('rocket', len(churn_per_cat)))
    axes[i].set_title(f"Churned Customers by {col}")

plt.tight_layout()
st.pyplot(fig5)

# ---------------- Tenure Group ----------------
churned['tenure_group'] = pd.cut(churned['tenure'], bins=[0,6,12,float('inf')],
                                 labels=['0-6 months','6-12 months','>12 months'])
st.write("Churned Customer Tenure Group:")
st.dataframe(churned['tenure_group'].value_counts())

# ---------------- Offers and Referrals ----------------
st.subheader("Offers and Referrals among Churned Customers")
result = (
    churned.groupby("offer")
    .apply(lambda g: pd.Series({
        "Total Customers": len(g),
        "With 3 Referrals": (g["number_of_referrals"] >= 3).sum(),
        "6 Months Tenure": (g["tenure"] >= 6).sum(),
        "+12 Months Tenure": (g["tenure"] >= 12).sum()
    }))
    .reset_index()
)
st.dataframe(result)

# ---------------- Churn Categories ----------------
st.subheader("Churn Categories")
st.dataframe(churned['churn_category'].value_counts())

st.subheader("Churn Reasons")
st.dataframe(churned['churn_reason'].value_counts().sort_values(ascending=False))

# ---------------- Location Analysis ----------------
st.subheader("Top Cities with Churned Customers")
st.dataframe(churned['city'].value_counts().head(10))
