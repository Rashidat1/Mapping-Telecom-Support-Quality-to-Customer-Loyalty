import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Set layout configurations
st.set_page_config(page_title="Telecom Retention Engine", page_icon="📱", layout="wide")

# --- 1. OPTIMIZED STANDALONE DATA INJECTION ---
# This ensures recruiters can click through your app online without raw CSV errors
@st.cache_data
def load_portfolio_data():
    np.random.seed(42)
    n_samples = 2800
    
    # Simulating your exact validated dataframe structure
    data = pd.DataFrame({
        'user_id': [f"TEL_{2000+i}" for i in range(n_samples)],
        'monthly_fee': np.random.uniform(40, 110, n_samples),
        'avg_weekly_usage_hours': np.random.uniform(2, 48, n_samples),
        'support_tickets': np.random.randint(0, 6, n_samples),
        'payment_failures': np.random.randint(0, 4, n_samples),
        'tenure_months': np.random.randint(1, 24, n_samples),
        'last_login_days_ago': np.random.randint(0, 30, n_samples),
        'average_csat_score': np.random.uniform(1.0, 5.0, n_samples),
        'average_ticket_sentiment': np.random.uniform(-0.6, 0.6, n_samples)
    })
    
    # Aligning churn logic to mirror your exact model drivers
    churn_score = (
        ((50 - data['avg_weekly_usage_hours']) * 0.015) +
        (data['last_login_days_ago'] * 0.02) +
        (data['payment_failures'] * 0.15) -
        (data['tenure_months'] * 0.01)
    )
    data['churn'] = np.random.binomial(1, np.clip(churn_score, 0.05, 0.95))
    return data

df_master = load_portfolio_data()

# --- 2. MODEL INTERNALS CONFIGURATION ---
features = [
    'avg_weekly_usage_hours', 'last_login_days_ago', 'tenure_months',
    'payment_failures', 'support_tickets', 'average_ticket_sentiment',
    'monthly_fee', 'average_csat_score'
]
X = df_master[features]
y = df_master['churn']
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced').fit(X, y)

# --- 3. STREAMLIT FRONT-END DASHBOARD UI ---
st.title("📱 Data-Driven Telecom Subscriber Retention Dashboard")
st.markdown("### **Portfolio Showcase** | *Evaluating Customer Experience Levers Against Subscription Churn*")
st.hr()

# Metric Cards Panel
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric("Total Subscribers Analyzed", f"{len(df_master):,}")
with m_col2:
    st.metric("Baseline Churn Rate", f"{(df_master['churn'].mean()):.2%}")
with m_col3:
    st.metric("Total Revenue Leaking from Churn", f"${df_master[df_master['churn'] == 1]['monthly_fee'].sum():,.2f}")

st.hr()

# Layout Splits: Simulator Control on Left, Diagnostic Card on Right
layout_left, layout_right = st.columns([2, 1])

with layout_left:
    st.subheader("🎛️ Executive Strategic Simulator")
    st.write("Adjust the behavioral levers below to simulate proactive retention actions and calculate top-line savings.")
    
    # Strategic Sliders
    sim_usage = st.slider("Target Minimum Weekly Platform Usage (Hours)", 0.0, 40.0, 5.0)
    sim_login = st.slider("Maximum Days Allowed Since Last Login (Intervention Gap)", 1, 30, 15)
    
    # Apply modifications to temporary simulated database
    df_sim = df_master.copy()
    df_sim['avg_weekly_usage_hours'] = np.where(df_sim['avg_weekly_usage_hours'] < sim_usage, sim_usage, df_sim['avg_weekly_usage_hours'])
    df_sim['last_login_days_ago'] = np.where(df_sim['last_login_days_ago'] > sim_login, sim_login, df_sim['last_login_days_ago'])
    
    # Run predictions against modifications
    sim_probs = clf.predict_proba(df_sim[features])[:, 1]
    saved_users = max(0, int(df_master['churn'].sum() - (sim_probs > 0.5).sum()))
    recovered_revenue = saved_users * df_master['monthly_fee'].mean() * 12
    
    # Display Simulated Results Box
    st.success(f"🎉 **Strategic Impact:** Intercepting users at these boundaries preserves **{saved_users} subscriber accounts**, saving **${recovered_revenue:,.2f} in Annualized Gross Revenue**.")

with layout_right:
    st.subheader("🔍 Account Risk Lookup")
    lookup_id = st.selectbox("Select Account ID to Profile:", df_master['user_id'].values)
    
    # Pull individual metadata card row
    ind_record = df_master[df_master['user_id'] == lookup_id].iloc[0]
    ind_features = pd.DataFrame([ind_record[features]])
    risk_metric = clf.predict_proba(ind_features)[0][1]
    
    # Layout account risk classification metrics visually
    st.markdown(f"**Customer File:** `{lookup_id}`")
    st.metric("ML Evaluated Churn Risk", f"{risk_metric:.1%}")
    if risk_metric > 0.65:
        st.error("🚨 Account Status: Critical Churn Threat")
    elif risk_metric > 0.40:
        st.warning("⚠️ Account Status: Elevated Dormancy Risk")
    else:
        st.success("✅ Account Status: Healthy Active Subscriber")
        
    st.markdown(f"""
    * **Weekly Usage Volume:** {ind_record['avg_weekly_usage_hours']:.1f} hours
    * **Inactivity Ingest Gap:** {int(ind_record['last_login_days_ago'])} days ago
    * **Total Support Interactions:** {int(ind_record['support_tickets'])} tickets
    * **Assigned CSAT Sentiment:** {ind_record['average_csat_score']:.1f}/5.0
    """)

st.hr()

# Bottom Layout Visual Panel Plots
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    st.subheader("Platform Usage Patterns vs Churn Density")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.kdeplot(data=df_master[df_master['churn'] == 0], x='avg_weekly_usage_hours', fill=True, color='#1f77b4', label='Retained', ax=ax, alpha=0.4)
    sns.kdeplot(data=df_master[df_master['churn'] == 1], x='avg_weekly_usage_hours', fill=True, color='#d62728', label='Churned', ax=ax, alpha=0.5)
    ax.set_xlabel("Weekly Usage (Hours)")
    ax.legend()
    st.pyplot(fig)

with plot_col2:
    st.subheader("Model Feature Importance Weights")
    importance_scores = clf.feature_importances_
    f_df = pd.DataFrame({'Feature': features, 'Weight': importance_scores}).sort_values('Weight', ascending=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(f_df['Feature'], f_df['Weight'], color='#2b5c8f', edgecolor='black', linewidth=0.5)
    st.pyplot(fig)
