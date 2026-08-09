# University Admission & Major Choice Dataset

## Overview

This dataset contains information about university admission preferences, university programs, majors, applicant exam scores, and major capacities.

The dataset can be used for:

- Exploratory Data Analysis (EDA)
- University and major preference analysis
- Applicant score analysis
- University popularity analysis
- Major popularity analysis
- Admission recommendation systems
- Classification and machine learning
- Business intelligence and decision support

The data is divided into four CSV files:

1. `majors.csv`
2. `universities.csv`
3. `score_humanities.csv`
4. `score_science.csv`

---

## Dataset Files

### 1. `majors.csv`

Contains information about available university majors and their capacities.

**Rows:** 3,167  
**Columns:** 6

| Column | Description |
|---|---|
| `id_major` | Unique identifier for the major |
| `id_university` | Identifier of the university offering the major |
| `type` | Major category: `science` or `humanities` |
| `major_name` | Name of the major |
| `capacity` | Number of available places/capacity for the major |
| `Unnamed: 0` | Original row index from the source dataset |

The `id_university` column can be linked to `universities.csv`.

---

### 2. `universities.csv`

Contains information about universities.

**Rows:** 85  
**Columns:** 3

| Column | Description |
|---|---|
| `id_university` | Unique identifier for the university |
| `university_name` | Name of the university |
| `Unnamed: 0` | Original row index from the source dataset |

This file can be joined with `majors.csv` using:

```text
majors.id_university = universities.id_university
```

---

### 3. `score_humanities.csv`

Contains applicant preferences and examination scores for humanities applicants.

**Rows:** 61,202  
**Columns:** 15

| Column | Description |
|---|---|
| `id_first_major` | Applicant's first-choice major |
| `id_first_university` | Applicant's first-choice university |
| `id_second_major` | Applicant's second-choice major |
| `id_second_university` | Applicant's second-choice university |
| `id_user` | Applicant identifier |
| `score_eko` | Economics score |
| `score_geo` | Geography score |
| `score_kmb` | General/competency-related score |
| `score_kpu` | Academic potential/competency score |
| `score_kua` | General academic score |
| `score_mat` | Mathematics score |
| `score_ppu` | General knowledge/understanding score |
| `score_sej` | History score |
| `score_sos` | Sociology score |
| `Unnamed: 0` | Original row index |

The humanities file contains the applicant's first and second program choices together with subject-level scores.

---

### 4. `score_science.csv`

Contains applicant preferences and examination scores for science applicants.

**Rows:** 86,570  
**Columns:** 14

| Column | Description |
|---|---|
| `id_first_major` | Applicant's first-choice major |
| `id_first_university` | Applicant's first-choice university |
| `id_second_major` | Applicant's second-choice major |
| `id_second_university` | Applicant's second-choice university |
| `id_user` | Applicant identifier |
| `score_bio` | Biology score |
| `score_fis` | Physics score |
| `score_kim` | Chemistry score |
| `score_kmb` | General/competency-related score |
| `score_kpu` | Academic potential/competency score |
| `score_kua` | General academic score |
| `score_mat` | Mathematics score |
| `score_ppu` | General knowledge/understanding score |
| `Unnamed: 0` | Original row index |

---

## Dataset Size

| File | Rows | Columns |
|---|---:|---:|
| `majors.csv` | 3,167 | 6 |
| `universities.csv` | 85 | 3 |
| `score_humanities.csv` | 61,202 | 15 |
| `score_science.csv` | 86,570 | 14 |

There are **150,939 applicant records** across the humanities and science score files.

The dataset contains **85 universities** and **3,167 major records**.

---

## Relationships Between Files

The main relationships are based on ID columns.

### University relationship

```text
universities
    |
    | id_university
    |
    +---- majors
```

The university ID connects university information with its available majors.

### Major relationship

```text
majors.id_major
       |
       +---- score_humanities.id_first_major
       |
       +---- score_humanities.id_second_major
       |
       +---- score_science.id_first_major
       |
       +---- score_science.id_second_major
```

This allows applicant choices to be connected to the corresponding major name and major capacity.

---

## Suggested Data Preparation

The `Unnamed: 0` columns are source/index columns and are generally not needed for analysis.

They can be removed with:

```python
df = df.drop(columns=["Unnamed: 0"])
```

Before combining or analyzing the data, it is also recommended to check:

```python
df.info()
df.isnull().sum()
df.duplicated().sum()
```

---

## Combining Science and Humanities Applicants

The two score datasets have different subject columns, so they should normally be analyzed separately for subject-level analysis.

For general applicant analysis, they can be combined after adding an `exam_type` column:

```python
science = pd.read_csv("score_science.csv")
humanities = pd.read_csv("score_humanities.csv")

science["exam_type"] = "science"
humanities["exam_type"] = "humanities"

scores = pd.concat(
    [science, humanities],
    ignore_index=True,
    sort=False
)
```

Because the science and humanities files contain different subjects, some subject columns will be missing (`NaN`) after concatenation. This is expected.

---

## Useful Analysis Ideas

### 1. Most Popular Majors

Count applicants by first-choice major:

```python
scores_final["first_major_name"].value_counts().head(10)
```

### 2. Most Popular Universities

Count applicants by first-choice university:

```python
scores_final["first_university_name"].value_counts().head(10)
```

### 3. Average Applicant Score

```python
scores_final["average_total_score"].mean()
```

### 4. Major Capacity Analysis

Compare applicant demand with available capacity:

```python
majors.groupby("major_name")["capacity"].sum()
```

### 5. First-Choice vs Second-Choice Analysis

Compare:

```text
id_first_major
id_second_major
```

to identify applicants whose first and second choices are identical or closely related.

---

## Business Insights

This dataset can support business and decision-making insights such as:

- Which majors attract the most applicants?
- Which universities are the most popular?
- What percentage of applicants are concentrated in the top 10 majors?
- What percentage of applicants choose the top 10 universities?
- How do high-scoring applicants distribute their choices?
- Which majors may have high demand relative to capacity?
- Which applicants may need safer second-choice options?
- Are science applicants choosing different universities or majors compared with humanities applicants?

---

## Machine Learning Applications

The dataset can also be used for machine learning projects.

Possible target variables include:

- First-choice major
- First-choice university
- Admission/recommendation category
- Major type

Possible input features include:

- Subject scores
- First-choice university
- First-choice major
- Second-choice university
- Second-choice major
- Exam type

Potential algorithms include:

- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- Gradient Boosting

Example imports:

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
```

---

## Recommendation Analysis

A recommendation system can classify applicants into categories based on their scores and choices.

Example categories:

```text
Diversify Choice: 1st & 2nd Majors Match
Consider Safety Major / Lower Cutoff Target
Target Top-Tier Flagship Universities
Recommend High-Demand STEM Track
Standard Application Track
```

These recommendations can be based on score thresholds, exam type, and applicant choices.

---

## Data Quality

A basic inspection of the supplied files found:

- No missing values in the four CSV files.
- No exact duplicate rows in the four CSV files.
- `majors.csv` contains both `science` and `humanities` major types.
- The score files represent two different applicant groups: science and humanities.
- The `Unnamed: 0` fields appear to be original dataframe/index columns.

---

## Python Requirements

A typical analysis environment requires:

```text
Python 3.x
pandas
numpy
matplotlib
scikit-learn
```

Install the packages with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Example Project Workflow

A recommended workflow is:

```text
1. Load the CSV files
        ↓
2. Clean unnecessary index columns
        ↓
3. Inspect missing values and duplicates
        ↓
4. Combine/link university and major information
        ↓
5. Calculate applicant scores
        ↓
6. Analyze first and second choices
        ↓
7. Visualize popular majors and universities
        ↓
8. Generate business insights
        ↓
9. Build applicant recommendations
        ↓
10. Train and evaluate machine learning models
```

---

## Notes

The dataset should be treated as an analytical dataset. Column abbreviations such as `score_kpu`, `score_kua`, `score_kmb`, and `score_ppu` are retained from the original data rather than renamed, because their exact definitions may depend on the original examination/data documentation.

When creating derived fields such as `average_total_score`, document the formula used so that results can be reproduced.

---

## License and Source

The supplied archive contains the four CSV files listed above. No license or original source documentation was included in the uploaded archive.

Before redistributing the dataset, verify the original dataset's license and attribution requirements.
