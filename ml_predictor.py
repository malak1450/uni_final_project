import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "university_match_model.pkl"


@st.cache_resource
def load_model_bundle():
    """Load the trained model bundle saved by train_model.py."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def render_university_match_finder(df: pd.DataFrame):
    """Render the full ML section: header, metrics, inputs, and prediction results."""
    st.subheader("🤖 ML University Match Finder — Random Forest Classifier")
    st.caption(
        "Enter a score and the model returns the universities most associated with that "
        "score level (1st-choice history), ranked by predicted match probability, plus each "
        "university's historical score range for context (reach / target / safety)."
    )

    bundle = load_model_bundle()

    if bundle is None:
        st.warning(
            f"⚠️ No trained model found at `{MODEL_PATH}`. Run `python3 train_model.py` "
            "in the project folder first, then restart the app."
        )
        return

    model = bundle["model"]
    exam_encoder = bundle["exam_encoder"]
    feature_cols = bundle["feature_cols"]
    top1_acc = bundle["top1_acc"]
    top3_acc = bundle["top3_acc"]
    uni_stats = bundle["uni_stats"]
    id_to_name = bundle["id_to_name"]

    m1, m2 = st.columns(2)
    m1.metric("Top-1 Accuracy", f"{top1_acc:.1%}")
    m2.metric("Top-3 Accuracy", f"{top3_acc:.1%}")

    st.markdown("**🔮 Find Universities for a Given Score**")

    in_col1, in_col2, in_col3 = st.columns([1, 1, 1])
    with in_col1:
        input_score = st.number_input(
            "Student Score:",
            min_value=float(df["average_total_score"].min()),
            max_value=float(df["average_total_score"].max()),
            value=float(df["average_total_score"].mean()),
            step=1.0
        )
    with in_col2:
        input_exam = st.selectbox("Exam Stream:", options=df["exam_type"].unique(), key="ml_exam_input")
    with in_col3:
        top_n = st.slider("Number of universities to show:", min_value=3, max_value=15, value=10)

    if st.button("Find Matching Universities"):
        input_row = pd.DataFrame([{
            "average_total_score": input_score,
            "exam_type_enc": exam_encoder.transform([input_exam])[0]
        }])[feature_cols]

        proba = model.predict_proba(input_row)[0]
        classes = model.classes_

        results = pd.DataFrame({
            "id_first_university": classes,
            "Match Probability": proba
        }).sort_values("Match Probability", ascending=False).head(top_n)

        results["University"] = results["id_first_university"].map(id_to_name)
        results = results.merge(
            uni_stats[["id_first_university", "min", "mean", "max"]],
            on="id_first_university", how="left"
        )

        def fit_label(row):
            if input_score < row["min"]:
                return "🔴 Reach"
            elif input_score <= row["mean"]:
                return "🟡 Target"
            else:
                return "🟢 Safety"

        results["Fit"] = results.apply(fit_label, axis=1)
        results["Match Probability"] = (results["Match Probability"] * 100).round(1).astype(str) + "%"
        results = results.rename(columns={"min": "Historical Min", "mean": "Historical Avg", "max": "Historical Max"})
        results["Historical Min"] = results["Historical Min"].round(1)
        results["Historical Avg"] = results["Historical Avg"].round(1)
        results["Historical Max"] = results["Historical Max"].round(1)

        st.markdown(f"**Universities matching a score of {input_score:.1f} ({input_exam}):**")
        st.dataframe(
            results[["University", "Match Probability", "Fit", "Historical Min", "Historical Avg", "Historical Max"]],
            width='stretch',
            hide_index=True
        )