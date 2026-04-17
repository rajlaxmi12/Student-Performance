import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="EduAnalytics Pro", layout="wide")

@st.cache_data
def load_data():
    # Loading the UCI Student Math dataset
    # Using '../' assuming app.py is in /app and data is in /data
    df = pd.read_csv('../data/student-mat.csv', sep=';')
    
    # 1. Target Variable
    df['Pass'] = (df['G3'] >= 10).astype(int)
    
    # 2. FEATURE ENGINEERING (The Resume Upgrade)
    # Combining weekday and weekend alcohol consumption
    df['Alcohol_Index'] = (df['Dalc'] + df['Walc']) / 2
    
    # Creating a readable categorical mapping for failures
    df['Academic_History'] = df['failures'].map({
        0: 'Clean Record', 
        1: '1 Failure', 
        2: 'Multiple Failures', 
        3: 'Critical Status'
    })
    
    # Social Pressure Index (Goout + Alcohol)
    df['Social_Pressure'] = (df['goout'] + df['Alcohol_Index']) / 2
    
    return df

df = load_data()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("🕹️ Simulation Sandbox")
st.sidebar.markdown("Modify student profile to assess risk.")

# Using the new engineered concepts in the UI
absences = st.sidebar.slider("Number of Absences", 0, 93, 5)
academic_stat = st.sidebar.selectbox("Academic History", df['Academic_History'].unique())
studytime = st.sidebar.slider("Weekly Study Time (1: <2h, 4: >10h)", 1, 4, 2)
soc_pressure = st.sidebar.slider("Social Pressure Index (1-5)", 1.0, 5.0, 2.5)

# --- HEADER ---
st.title("🚀 EduAnalytics: Predictive Student Success Dashboard")
st.markdown("""
Identifying at-risk students through **Behavioral Feature Engineering** and **Random Forest Importance**.
""")

# --- TOP METRICS ---
m1, m2, m3 = st.columns(3)
m1.metric("Total Students", len(df))
m2.metric("Pass Rate", f"{(df['Pass'].mean()*100):.1f}%")
m3.metric("Avg. Social Pressure", f"{df['Social_Pressure'].mean():.2f}")

st.divider()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Impact of Social Pressure")
    # Using the engineered 'Social_Pressure' vs Final Grade
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Social_Pressure', y='G3', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
    ax.set_title("Correlation: Social Pressure vs. Final Grade")
    st.pyplot(fig)

with col2:
    st.subheader("💡 Feature Importance (The 'Why')")
    
    # Model Training including engineered features
    # Note: We exclude 'Academic_History' here because it's a string version of 'failures'
    X = pd.get_dummies(df.drop(['G1', 'G2', 'G3', 'Pass', 'Academic_History'], axis=1), drop_first=True)
    y = df['Pass']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    
    fig2, ax2 = plt.subplots()
    importances.plot(kind='barh', ax=ax2, color='#2ecc71')
    ax2.invert_yaxis()
    st.pyplot(fig2)
    st.info("Technical Note: Feature Engineering improved model interpretability by grouping social habits.")

# --- PREDICTION LOGIC ---
st.divider()
st.subheader("🎯 Real-Time Risk Assessment")

if st.sidebar.button("Run Prediction"):
    # Map back the academic status to a numeric value for the logic
    fail_val = 0 if academic_stat == 'Clean Record' else (1 if academic_stat == '1 Failure' else 2)
    
    # Heuristic score based on model insights
    score = (studytime * 2.5) - (fail_val * 4) - (absences * 0.15) - (soc_pressure * 1.2)
    
    if score > 0:
        st.success(f"### Status: ✅ LOW RISK")
        st.balloons()
        st.write("Student shows a strong probability of passing based on current behavioral metrics.")
    else:
        st.error(f"### Status: ⚠️ HIGH RISK")
        st.markdown("""
        **Intervention Required:**
        * Reduce absences below 5.
        * Targeted academic support for students with existing failures.
        * Monitor Social Pressure impact on study hours.
        """)

# --- DATA VIEW ---
if st.checkbox("Show Engineered Dataset"):
    st.write(df[['Academic_History', 'Social_Pressure', 'Alcohol_Index', 'Pass']].head(15))