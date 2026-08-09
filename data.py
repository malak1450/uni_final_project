import pandas as pd

def load_and_process_data(data_dir: str = "."):
    """Load, merge, and enrich the UTBK CSVs into a single scores DataFrame."""
    majors = pd.read_csv(f"{data_dir}/majors.csv")
    universities = pd.read_csv(f"{data_dir}/universities.csv")
    score_science = pd.read_csv(f"{data_dir}/score_science.csv")
    score_humanities = pd.read_csv(f"{data_dir}/score_humanities.csv")

    score_science["exam_type"] = "science"
    score_humanities["exam_type"] = "humanities"
    scores = pd.concat([score_science, score_humanities], ignore_index=True)

    major_id_col = "id_major" if "id_major" in majors.columns else "id"
    scores = scores.merge(
        majors[[major_id_col, "major_name"]].rename(
            columns={major_id_col: "id_first_major", "major_name": "first_major_name"}
        ),
        on="id_first_major", how="left"
    )
    scores = scores.merge(
        majors[[major_id_col, "major_name"]].rename(
            columns={major_id_col: "id_second_major", "major_name": "second_major_name"}
        ),
        on="id_second_major", how="left"
    )

    uni_id_col = "id_university" if "id_university" in universities.columns else "id"
    scores = scores.merge(
        universities[[uni_id_col, "university_name"]].rename(
            columns={uni_id_col: "id_first_university", "university_name": "first_university_name"}
        ),
        on="id_first_university", how="left"
    )
    scores = scores.merge(
        universities[[uni_id_col, "university_name"]].rename(
            columns={uni_id_col: "id_second_university", "university_name": "second_university_name"}
        ),
        on="id_second_university", how="left"
    )

    numeric_cols = scores.select_dtypes(include='number').columns
    score_cols = [col for col in numeric_cols if not col.startswith('id_') and col != 'id']
    scores['average_total_score'] = scores[score_cols].mean(axis=1)

    avg_score = scores["average_total_score"].mean()
    high_score_threshold = scores["average_total_score"].quantile(0.75)
    low_score_threshold = scores["average_total_score"].quantile(0.25)

    def recommend(row):
        if row["id_first_major"] == row["id_second_major"]:
            return "Diversify Choice: 1st & 2nd Majors Match"
        elif row["average_total_score"] < low_score_threshold:
            return "Consider Safety Major / Lower Cutoff Target"
        elif row["average_total_score"] >= high_score_threshold:
            return "Target Top-Tier Flagship Universities"
        elif row["exam_type"] == "science" and row["average_total_score"] > avg_score:
            return "Recommend High-Demand STEM Track"
        else:
            return "Standard Application Track"

    scores["Recommendation"] = scores.apply(recommend, axis=1)
    return scores