import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data import load_and_process_data
from ml_predictor import render_university_match_finder

st.set_page_config(
    page_title="UTBK Admissions & Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 UTBK Student Performance & Admissions Dashboard")
st.caption("Comprehensive Analysis: KPIs, Major/University Demand, Business Insights & Strategic Recommendations")

@st.cache_data
def get_data():
    return load_and_process_data()

df = get_data()

st.sidebar.header("🔍 Global Filters")

exam_filter = st.sidebar.multiselect(
    "Exam Stream:",
    options=df["exam_type"].unique(),
    default=df["exam_type"].unique()
)

min_score, max_score = float(df["average_total_score"].min()), float(df["average_total_score"].max())
score_range = st.sidebar.slider(
    "Score Range Filter:",
    min_value=round(min_score, 1),
    max_value=round(max_score, 1),
    value=(round(min_score, 1), round(max_score, 1))
)

filtered_df = df[
    (df["exam_type"].isin(exam_filter)) &
    (df["average_total_score"] >= score_range[0]) &
    (df["average_total_score"] <= score_range[1])
]

st.subheader("📌 Summary Metrics & Calculated KPIs")

total_students = len(filtered_df)
science_pct = (filtered_df["exam_type"] == "science").sum() / total_students * 100 if total_students > 0 else 0
humanities_pct = (filtered_df["exam_type"] == "humanities").sum() / total_students * 100 if total_students > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Applicants", f"{total_students:,}")
kpi2.metric("Science / Humanities Split", f"{science_pct:.1f}% / {humanities_pct:.1f}%")
kpi3.metric("Overall Average Score", f"{filtered_df['average_total_score'].mean():.2f}" if total_students > 0 else "0")
kpi4.metric("Highest Score Achieved", f"{filtered_df['average_total_score'].max():.2f}" if total_students > 0 else "0")
kpi5.metric("Unique 1st Majors", filtered_df["id_first_major"].nunique())

st.markdown("---")

st.subheader("💡 Automated Business Insights")

if total_students > 0:
    top_10_majors_pct = (filtered_df["first_major_name"].value_counts().head(10).sum() / total_students) * 100
    top_10_unis_pct = (filtered_df["first_university_name"].value_counts().head(10).sum() / total_students) * 100
    global_avg = filtered_df["average_total_score"].mean()
    p90_score = filtered_df["average_total_score"].quantile(0.90)

    insights_data = [
        {
            "Insight Category": "High Major Demand Concentration",
            "Key Value": f"{top_10_majors_pct:.1f}%",
            "Strategic Context": "Share of all 1st-choice applications captured by the top 10 majors."
        },
        {
            "Insight Category": "Institutional Brand Dominance",
            "Key Value": f"{top_10_unis_pct:.1f}%",
            "Strategic Context": "Share of primary applicants attracted by the top 10 universities."
        },
        {
            "Insight Category": "Score Selectivity Gap",
            "Key Value": f"{p90_score:.1f} vs {global_avg:.1f}",
            "Strategic Context": "Top 10th percentile score threshold vs. global average applicant score."
        }
    ]
    st.table(pd.DataFrame(insights_data))
else:
    st.warning("No data matches current filter settings.")

st.markdown("---")

st.subheader("📊 Applicant Distributions & Preferences")

col_left, col_mid, col_right = st.columns([1, 1.2, 1.2])

with col_left:
    st.markdown("**Participant Split: Science vs Humanities**")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    if total_students > 0:
        filtered_df["exam_type"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            colors=["#4C72B0", "#DD8452"],
            startangle=140,
            ax=ax1
        )
        ax1.set_ylabel("")
    st.pyplot(fig1)

with col_mid:
    st.markdown("**Top 10 Most Popular 1st Choice Majors**")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    if total_students > 0:
        filtered_df["first_major_name"].value_counts().head(10).sort_values().plot(
            kind="barh", color="#55A868", ax=ax2
        )
        ax2.set_xlabel("Applicants")
    st.pyplot(fig2)

with col_right:
    st.markdown("**Top 10 Most Popular 1st Choice Universities**")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    if total_students > 0:
        filtered_df["first_university_name"].value_counts().head(10).sort_values().plot(
            kind="barh", color="#C44E52", ax=ax3
        )
        ax3.set_xlabel("Applicants")
    st.pyplot(fig3)

st.markdown("---")

st.subheader("🎯 Student Recommendation Engine Summary")

rec_col1, rec_col2 = st.columns([1, 2])

with rec_col1:
    st.markdown("**Action Category Counts**")
    rec_counts = filtered_df["Recommendation"].value_counts().reset_index()
    rec_counts.columns = ["Recommendation Action", "Count"]
    st.dataframe(rec_counts, width='stretch', hide_index=True)

with rec_col2:
    st.markdown("**Recommendation Category Distribution**")
    st.bar_chart(filtered_df["Recommendation"].value_counts())

st.markdown("---")

render_university_match_finder(df)

st.markdown("---")

st.markdown("📋 Filtered Applicant Dataset Preview")

display_cols = [
    "exam_type", "first_major_name", "first_university_name",
    "second_major_name", "second_university_name",
    "average_total_score", "Recommendation"
]

st.dataframe(
    filtered_df[display_cols].head(200),
    width='stretch'
)

csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Results as CSV",
    data=csv_data,
    file_name="utbk_applicant_recommendations.csv",
    mime="text/csv"
)