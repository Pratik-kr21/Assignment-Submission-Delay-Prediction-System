import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EduPredict | Submission Delay Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model.joblib"
DATA_PATH = "data.csv"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        from train import train
        model, _ = train(data_path=DATA_PATH, model_output_path=MODEL_PATH)
        return model
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    from train import load_data
    return load_data(DATA_PATH)


def evaluate_risk(prob: float) -> tuple[str, str]:
    if prob >= 0.75:
        return "High", "Immediate instructor intervention & automated email nudge recommended."
    elif prob >= 0.40:
        return "Moderate", "Friendly study reminder & office hours alert suggested."
    else:
        return "Low", "Student on track. No action needed."


def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    probs = model.predict_proba(df)[:, 1]
    res["delay_probability"] = np.round(probs, 4)
    res["is_late_pred"] = (probs >= 0.5).astype(int)
    tiers = []
    actions = []
    for p in probs:
        t, a = evaluate_risk(p)
        tiers.append(t)
        actions.append(a)
    res["risk_tier"] = tiers
    res["recommended_action"] = actions
    return res


model = load_model()

st.sidebar.title("🎓 EduPredict AI")
st.sidebar.markdown("**Assignment Submission Delay Prediction**")
view_mode = st.sidebar.radio(
    "Choose View",
    ["👩‍🏫 Instructor Dashboard", "🎒 Student Self-Check Portal"],
)

st.sidebar.info("💡 **Model**: XGBoost Classification Pipeline\n- Features: VLE Clicks, Past Delays, Deadline Density")

if view_mode == "👩‍🏫 Instructor Dashboard":
    st.title("👩‍🏫 Instructor Delay Early-Warning Dashboard")
    st.markdown("Monitor student cohorts before deadlines to identify students needing intervention.")

    cohort_full = load_dataset()
    sample_cohort = cohort_full.head(100).copy()
    predictions = predict_df(model, sample_cohort)

    total_students = len(predictions)
    high_count = (predictions["risk_tier"] == "High").sum()
    mod_count = (predictions["risk_tier"] == "Moderate").sum()
    low_count = (predictions["risk_tier"] == "Low").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Enrolled Cohort", f"{total_students} Students")
    col2.metric("⚠️ High Delay Risk", f"{high_count}", f"{(high_count / total_students * 100):.1f}%")
    col3.metric("⏳ Moderate Risk", f"{mod_count}")
    col4.metric("✅ On Track", f"{low_count}")

    st.markdown("---")

    st.subheader("📋 Cohort Risk Assessment")
    filter_tiers = st.multiselect(
        "Filter by Risk Tier",
        ["High", "Moderate", "Low"],
        default=["High", "Moderate"],
    )

    filtered = predictions[predictions["risk_tier"].isin(filter_tiers)].sort_values(
        by="delay_probability", ascending=False
    )

    display_cols = [
        "student_id",
        "delay_probability",
        "risk_tier",
        "clicks_last_7_days",
        "hist_avg_delay_days",
        "concurrent_deadlines_window",
        "assessment_type",
        "assessment_weight",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True)

    st.markdown("---")
    st.subheader("📬 Automated Early-Intervention Assistant")
    high_risk_list = predictions[predictions["risk_tier"] == "High"]["student_id"].tolist()

    if high_risk_list:
        selected_student = st.selectbox("Select high-risk student for outreach:", high_risk_list)
        student_data = predictions[predictions["student_id"] == selected_student].iloc[0]

        prob_pct = int(student_data["delay_probability"] * 100)
        email_draft = f"""Subject: Checking in on your upcoming {student_data['assessment_type']} assignment

Dear {selected_student},

I hope your week is going well. I'm reaching out as we noticed you have upcoming assessment deadlines and may have competing course commitments this week.

Our student support system flagged that you might benefit from additional guidance on the upcoming {student_data['assessment_type']} (Weight: {student_data['assessment_weight']}%). 

Please feel free to book a 1-on-1 drop-in slot or reach out during office hours this week so we can review the prompt together and ensure you stay on schedule.

Best regards,
Course Instructor & Student Support Team
"""
        st.text_area("Generated Outreach Message:", email_draft, height=220)
        st.success(f"Nudge template prepared for {selected_student} (Delay Risk: {prob_pct}%).")
    else:
        st.info("No high-risk students currently identified in this cohort.")

else:
    st.title("🎒 Student Submission Self-Check Portal")
    st.markdown("Check your assignment delay risk 3–5 days prior to the deadline and receive personalized tips.")

    with st.form("student_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### 📚 Academic Context")
            hist_avg = st.slider("Past Average Submission Delay (days)", -5.0, 10.0, 0.5, step=0.1)
            hist_rate = st.slider("Past Late Submission Rate (%)", 0, 100, 20) / 100.0
            hist_max = st.slider("Past Maximum Delay (days)", 0.0, 15.0, 1.5, step=0.1)
            assessment_type = st.selectbox("Assessment Type", ["TMA", "CMA", "Exam"])
            assessment_weight = st.slider("Assessment Weight (% of grade)", 5.0, 60.0, 20.0, step=5.0)

        with col_right:
            st.markdown("##### 💻 Study Activity & Workload")
            clicks_7 = st.number_input("LMS Activity (clicks last 7 days)", min_value=0, max_value=500, value=35)
            clicks_14 = st.number_input("LMS Activity (clicks last 14 days)", min_value=clicks_7, max_value=1000, value=90)
            concurrent = st.slider("Other Deadlines within ±3 days", 0, 5, 1)
            studied_credits = st.selectbox("Total Credits Enrolled", [30, 60, 90, 120], index=1)
            days_to_first = st.slider("Days to First Access Course Materials", 0.0, 20.0, 3.0, step=0.5)

        submit_btn = st.form_submit_button("🔍 Calculate Submission Delay Risk", use_container_width=True)

    if submit_btn:
        decay_ratio = round((clicks_7 + 1) / (clicks_14 - clicks_7 + 1), 3)

        input_data = pd.DataFrame([{
            "hist_avg_delay_days": hist_avg,
            "hist_late_rate": hist_rate,
            "hist_max_delay_days": hist_max,
            "clicks_last_7_days": clicks_7,
            "clicks_last_14_days": clicks_14,
            "activity_decay_ratio": decay_ratio,
            "days_to_first_vle_access": days_to_first,
            "concurrent_deadlines_window": concurrent,
            "studied_credits": studied_credits,
            "assessment_weight": assessment_weight,
            "assessment_type": assessment_type,
        }])

        prob = float(model.predict_proba(input_data)[0, 1])
        tier, action = evaluate_risk(prob)

        st.markdown("---")
        st.subheader("📊 Your Assessment Timeliness Forecast")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Predicted Delay Probability", f"{prob * 100:.1f}%")
            if tier == "High":
                st.error("🚨 Risk Status: HIGH DELAY RISK")
            elif tier == "Moderate":
                st.warning("⚠️ Risk Status: MODERATE DELAY RISK")
            else:
                st.success("✅ Risk Status: ON TRACK (LOW RISK)")

        with res_col2:
            st.markdown("##### 📌 Guidance & Recommended Next Steps:")
            st.info(action)
            if prob >= 0.5:
                st.markdown("""
                - 🎯 **Break down tasks**: Divide the assignment into 3 sub-tasks to be completed over the next 48 hours.
                - 📅 **Buffer deadline**: Plan to complete the draft 24 hours before the official cutoff.
                - 🙋‍♂️ **Ask early**: If stuck, message your TA or tutor immediately.
                """)
            else:
                st.markdown("""
                - 🌟 **Keep up momentum**: Your engagement velocity is strong.
                - 🔍 **Final Review**: Double check assignment guidelines and rubric before final submission.
                """)
