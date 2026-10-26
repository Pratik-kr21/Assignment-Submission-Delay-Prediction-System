"""Streamlit Dashboard for Assignment Submission Delay Prediction System."""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict_model import DelayPredictor
from src.data.make_dataset import generate_synthetic_data
from src.monitoring.drift_detector import generate_drift_summary

st.set_page_config(
    page_title="Submission Delay Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-high { color: #d32f2f; font-weight: bold; }
    .badge-mod { color: #f57c00; font-weight: bold; }
    .badge-low { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    model_path = os.path.join("models", "model.joblib")
    if not os.path.exists(model_path):
        from src.models.train_model import train
        train(model_type="xgboost")
    return DelayPredictor(model_path=model_path)


st.sidebar.title("🎓 EduPredict MLOps")
st.sidebar.markdown("**Assignment Submission Delay Predictor**")
portal_mode = st.sidebar.radio("Navigation", ["👩‍🏫 Instructor Dashboard", "🎒 Student Self-Check Portal", "📊 MLOps & Drift Observability"])

predictor = get_predictor()

# ==========================================================
# 1. INSTRUCTOR DASHBOARD
# ==========================================================
if portal_mode == "👩‍🏫 Instructor Dashboard":
    st.title("👩‍🏫 Course Instructor Delay Early-Warning Dashboard")
    st.markdown("Proactively monitor assignment submission timeliness and trigger automated support interventions.")

    # Generate benchmark cohort for demonstration
    @st.cache_data
    def load_cohort():
        df = generate_synthetic_data(num_samples=120)
        return df

    cohort_df = load_cohort()
    predictions_df = predictor.predict_batch(cohort_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Enrolled Students", len(predictions_df))
    with col2:
        high_risk = (predictions_df["risk_tier"] == "High").sum()
        st.metric("⚠️ High Delay Risk", high_risk, delta=f"{(high_risk/len(predictions_df)*100):.1f}%")
    with col3:
        mod_risk = (predictions_df["risk_tier"] == "Moderate").sum()
        st.metric("⏳ Moderate Risk", mod_risk)
    with col4:
        low_risk = (predictions_df["risk_tier"] == "Low").sum()
        st.metric("✅ On Track", low_risk)

    st.markdown("---")
    st.subheader("📋 Student Risk Assessment Table")
    
    tier_filter = st.multiselect("Filter by Risk Tier", ["High", "Moderate", "Low"], default=["High", "Moderate"])
    filtered_df = predictions_df[predictions_df["risk_tier"].isin(tier_filter)]
    
    display_cols = [
        "student_id", "delay_probability", "risk_tier",
        "clicks_last_7_days", "hist_avg_delay_days", "concurrent_deadlines_window", "assessment_weight"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values(by="delay_probability", ascending=False),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("📬 Automated Early-Intervention Assistant")
    high_risk_students = predictions_df[predictions_df["risk_tier"] == "High"]
    if len(high_risk_students) > 0:
        selected_student = st.selectbox("Select high-risk student to draft alert:", high_risk_students["student_id"].tolist())
        student_data = high_risk_students[high_risk_students["student_id"] == selected_student].iloc[0]
        
        email_template = f"""Subject: Academic Support Check-in: Upcoming {student_data['assessment_type']} Assignment

Dear {selected_student},

We noticed that the upcoming assignment is due in 3 days, and your LMS portal activity shows only {student_data['clicks_last_7_days']} clicks this week alongside {student_data['concurrent_deadlines_window']} other concurrent deadlines.

We want to make sure you have all the resources you need to succeed. Our Teaching Assistants have office hours open tomorrow from 2:00 PM - 4:00 PM. 

Please don't hesitate to reach out if you need clarification or guidance!

Best regards,
Course Instruction Team"""
        st.text_area("Generated Pedagogical Nudge Email:", email_template, height=220)
        if st.button("Send Automated Alert"):
            st.success(f"Nudge email dispatched to {selected_student}!")

# ==========================================================
# 2. STUDENT SELF-CHECK PORTAL
# ==========================================================
elif portal_mode == "🎒 Student Self-Check Portal":
    st.title("🎒 Student Submission Readiness & Time Management")
    st.markdown("Evaluate your submission timeline risk and receive personalized study planning tips.")

    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            hist_delay = st.slider("Past average submission delay (days)", min_value=-5.0, max_value=10.0, value=0.0, step=0.5)
            hist_late_rate = st.slider("Percentage of previous assignments submitted late", 0.0, 1.0, 0.2, 0.05)
            clicks_7d = st.number_input("LMS page views/downloads in the last 7 days", min_value=0, max_value=300, value=25)
            clicks_14d = st.number_input("Total LMS page views in the last 14 days", min_value=0, max_value=600, value=60)
        with col2:
            days_started = st.slider("Days after release you started working on this", 0.0, 15.0, 2.0, 0.5)
            concurrent = st.number_input("Other exams/assignments due in the same week", min_value=0, max_value=10, value=1)
            studied_credits = st.selectbox("Enrolled Credits", [30, 60, 90, 120], index=1)
            weight = st.slider("Assignment Weight (% of final grade)", 5.0, 50.0, 20.0, 5.0)
            assessment_type = st.selectbox("Assessment Format", ["TMA", "CMA", "Exam"])

        submit_btn = st.form_submit_button("Analyze My Submission Risk")

    if submit_btn:
        decay = (clicks_7d + 1) / (max(clicks_14d - clicks_7d, 0) + 1)
        input_data = {
            "hist_avg_delay_days": hist_delay,
            "hist_late_rate": hist_late_rate,
            "hist_max_delay_days": max(hist_delay, 1.0),
            "clicks_last_7_days": clicks_7d,
            "clicks_last_14_days": clicks_14d,
            "activity_decay_ratio": round(decay, 3),
            "days_to_first_vle_access": days_started,
            "concurrent_deadlines_window": concurrent,
            "studied_credits": studied_credits,
            "assessment_weight": weight,
            "assessment_type": assessment_type
        }
        res = predictor.predict_single(input_data)
        
        st.markdown("---")
        st.subheader("Your Submission Timeline Forecast")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("Predicted Delay Probability", f"{res['delay_probability']*100:.1f}%")
            if res["risk_tier"] == "High":
                st.error("⚠️ Status: High Risk of Delay")
            elif res["risk_tier"] == "Moderate":
                st.warning("⏳ Status: Moderate Timeline Tightness")
            else:
                st.success("✅ Status: On Track for Early/On-Time Submission")
        with col_r2:
            st.info(f"**Action Plan**: {res['recommended_action']}")
            if res["delay_probability"] > 0.4:
                st.markdown("""
                **Recommendations to stay on schedule:**
                - 📅 Block a 2-hour uninterrupted sprint today.
                - 🤝 Attend today's lab / TA office hours to resolve pending questions early.
                - 📝 Break the remaining assignment tasks into sub-deliverables.
                """)

# ==========================================================
# 3. MLOPS & DRIFT OBSERVABILITY
# ==========================================================
else:
    st.title("📊 MLOps Model Performance & Drift Observability")
    st.markdown("Live monitoring of production data drift, model metrics, and feature stability.")

    ref_data = generate_synthetic_data(num_samples=1000, random_seed=42)
    # Simulate current semester data with reduced engagement (drift)
    curr_data = generate_synthetic_data(num_samples=300, random_seed=99)
    curr_data["clicks_last_7_days"] = np.maximum(0, curr_data["clicks_last_7_days"] - 15)

    drift_report = generate_drift_summary(ref_data, curr_data)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Monitored Features", drift_report["features_analyzed"])
    with m2:
        st.metric("Drifted Features Detected", drift_report["drifted_features_count"])
    with m3:
        status = "🔴 Retrain Alert Triggered" if drift_report["dataset_drift_detected"] else "🟢 Production Stable"
        st.metric("Pipeline Health", status)

    st.subheader("Feature Drift Statistical Breakdown (KS-Test)")
    drift_table = []
    for feat, stats in drift_report["details"].items():
        drift_table.append({
            "Feature": feat,
            "KS Statistic": stats["ks_statistic"],
            "P-Value": stats["p_value"],
            "Drift Detected (p < 0.05)": "⚠️ DRIFT" if stats["drift_detected"] else "✅ OK"
        })
    st.dataframe(pd.DataFrame(drift_table), use_container_width=True)
