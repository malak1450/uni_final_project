import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
 
from data import load_and_process_data
 
MIN_SAMPLES_PER_UNI = 20
MODEL_PATH = "university_match_model.pkl"
 
 
def train_and_save_model():
    print("Loading and processing data...")
    df = load_and_process_data()
 
    ml_df = df[["exam_type", "average_total_score", "id_first_university", "first_university_name"]].dropna()
 
    counts = ml_df["id_first_university"].value_counts()
    valid_unis = counts[counts >= MIN_SAMPLES_PER_UNI].index
    ml_df = ml_df[ml_df["id_first_university"].isin(valid_unis)].copy()
 
    if ml_df["id_first_university"].nunique() < 2:
        raise ValueError(
            f"Not enough universities have >= {MIN_SAMPLES_PER_UNI} applicants to train a model. "
            "Lower MIN_SAMPLES_PER_UNI or add more data."
        )
 
    print(f"Training on {len(ml_df)} records across {ml_df['id_first_university'].nunique()} universities...")
 
    exam_encoder = LabelEncoder()
    ml_df["exam_type_enc"] = exam_encoder.fit_transform(ml_df["exam_type"])
 
    feature_cols = ["average_total_score", "exam_type_enc"]
    X = ml_df[feature_cols]
    y = ml_df["id_first_university"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    model = RandomForestClassifier(
        n_estimators=100,      
        max_depth=12,         
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    top1_acc = accuracy_score(y_test, y_pred)
 
    proba = model.predict_proba(X_test)
    classes = model.classes_
    top3_hits = 0
    for i, true_label in enumerate(y_test.values):
        top3_idx = proba[i].argsort()[-3:]
        top3_classes = classes[top3_idx]
        if true_label in top3_classes:
            top3_hits += 1
    top3_acc = top3_hits / len(y_test)
 
    print(f"Top-1 Accuracy: {top1_acc:.1%}")
    print(f"Top-3 Accuracy: {top3_acc:.1%}")
 
    uni_stats = ml_df.groupby(["id_first_university", "first_university_name"])["average_total_score"].agg(
        ["min", "mean", "max", "count"]
    ).reset_index()
 
    id_to_name = ml_df.drop_duplicates("id_first_university").set_index("id_first_university")["first_university_name"]
 
    bundle = {
        "model": model,
        "exam_encoder": exam_encoder,
        "feature_cols": feature_cols,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "uni_stats": uni_stats,
        "id_to_name": id_to_name,
        "min_samples_per_uni": MIN_SAMPLES_PER_UNI,
    }
 
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Model saved to {MODEL_PATH}")
 
 
if __name__ == "__main__":
    train_and_save_model()