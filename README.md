# Student Outcome Classifier Lab

Predicting student dropout and academic success with five classical ML classifiers, served
through an interactive Streamlit app.

- **GitHub repository:** `<add GitHub repository URL>`
- **Live Streamlit app:** `<add Streamlit Community Cloud URL>`

---

## a. Problem statement

Higher-education institutions lose students every year, and by the time a dropout is visible
it is usually too late to intervene. The goal of this project is to predict, from information
available at enrolment and from the first two semesters, whether a student will eventually
**drop out**, remain **enrolled** beyond the normal duration, or **graduate**. This is a
3-class classification problem.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive
Bayes and Support Vector Machine) are trained on the same data, evaluated with six metrics,
and deployed in a Streamlit app. The app scores an uploaded test CSV with the saved models
and does not retrain them.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository, dataset #697: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| DOI / licence | 10.24432/C5MC89 · CC BY 4.0 |
| Creators | Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021), Instituto Politécnico de Portalegre |
| Instances | 4,424 students (requirement ≥ 500) |
| Features | 36 (requirement ≥ 12) · real, integer and integer-coded categorical |
| Missing values | None |
| Target (`Target`) | `Graduate` 2,209 (49.9%) · `Dropout` 1,421 (32.1%) · `Enrolled` 794 (17.9%) |

Per the UCI page, the data was built from several disjoint databases of a higher-education
institution and covers students in degrees such as agronomy, design, education, nursing,
journalism, management, social service and technologies. The page also notes a strong
imbalance towards one class.

**Feature groups (36):**

- *Demographic:* marital status, nationality, gender, age at enrolment, displaced,
  international, educational special needs.
- *Socio-economic:* mother's / father's qualification and occupation, debtor, tuition fees up
  to date, scholarship holder.
- *Application / academic background:* application mode and order, course, day/evening
  attendance, previous qualification and its grade, admission grade.
- *Academic performance:* curricular units credited, enrolled, evaluated, approved, average
  grade and without evaluations, for both the 1st and 2nd semester (12 features).
- *Macro-economic:* unemployment rate, inflation rate, GDP.

**Preparation**

- Column names stripped of stray whitespace. One raw header contains a tab character.
- Stratified 80/20 split with `random_state=42`: 3,539 rows for training and 885 rows held
  out as `test_data.csv`. The test file keeps all 36 features and the `Target` column.
- Nine integer-coded nominal columns (e.g. *Course*, *Application mode*, *Mother's
  occupation*) are one-hot encoded. Codes seen fewer than 10 times are pooled into one
  "infrequent" column. All other columns are standardised. Preprocessing lives inside each
  saved scikit-learn `Pipeline`, so the app applies exactly the training-time transform.
  The Decision Tree uses raw features because trees are insensitive to scaling.
- Each model's hyper-parameters were chosen by 5-fold stratified grid search on the training
  split, scored on weighted F1.

## c. GitHub repository link

`<add GitHub repository URL>`

```
ML-Assignment/
├── app.py                     # Streamlit application
├── train_models.py            # Training, tuning, evaluation and model export
├── model_utils.py             # Shared constants and metric computation
├── requirements.txt
├── README.md
├── test_data.csv              # 885-row stratified hold-out set (features + Target)
├── data/
│   └── students_dropout_academic_success.csv   # raw UCI file (';'-separated)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── model_metadata.json    # feature order, class labels, tuned hyper-parameters
│   └── metrics_holdout.json   # metrics of every model on test_data.csv
└── .streamlit/config.toml     # app theme
```

## d. Models used

All five models are trained on the same data and evaluated on the same 885-row hold-out set
(`test_data.csv`). Because the task has three classes, Precision, Recall, F1 and AUC
(one-vs-rest) are **support-weighted averages**. MCC uses its multi-class form.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7740** | **0.8996** | **0.7612** | **0.7740** | **0.7636** | **0.6253** |
| Decision Tree | 0.7424 | 0.8698 | 0.7343 | 0.7424 | 0.7350 | 0.5737 |
| KNN | 0.7164 | 0.8531 | 0.7045 | 0.7164 | 0.6941 | 0.5275 |
| Naive Bayes | 0.6893 | 0.8302 | 0.6881 | 0.6893 | 0.6853 | 0.4879 |
| Support Vector Machine | 0.7672 | 0.8982 | 0.7561 | 0.7672 | 0.7576 | 0.6141 |

Source: `model/metrics_holdout.json`, produced by `python train_models.py`.

**Tuned hyper-parameters:**

| Model | Selected configuration |
|---|---|
| Logistic Regression | `C=1.0`, `class_weight=None` |
| Decision Tree | `criterion=entropy`, `max_depth=6`, `min_samples_leaf=20` |
| KNN | `n_neighbors=11`, `p=1` (Manhattan), `weights=uniform` |
| Naive Bayes | GaussianNB, `var_smoothing=0.1` |
| Support Vector Machine | RBF kernel, `C=5.0`, `gamma=0.01` |

**Per-class recall on the hold-out set** (supports the observations below):

| Model | Dropout (284) | Enrolled (159) | Graduate (442) |
|---|---|---|---|
| Logistic Regression | 0.754 | 0.409 | 0.919 |
| Decision Tree | 0.697 | 0.409 | 0.891 |
| KNN | 0.620 | 0.264 | 0.941 |
| Naive Bayes | 0.641 | 0.352 | 0.842 |
| Support Vector Machine | 0.739 | 0.409 | 0.914 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best on all six metrics (Accuracy 0.7740, AUC 0.8996, MCC 0.6253). It also has the highest Dropout recall (0.754), which is the class that matters most for early intervention. With one-hot encoded categories and standardised numeric features, the classes appear close to linearly separable: the RBF SVM did not beat it. Enabling `class_weight="balanced"` did not improve weighted F1 in cross-validation, so the unweighted model was kept. |
| **Decision Tree** | Mid-table (Accuracy 0.7424, MCC 0.5737), about 3 points below Logistic Regression. The tuned tree is shallow (depth 6, 42 leaves) and highly interpretable. *Curricular units 2nd sem (approved)* alone accounts for 67% of its feature importance, followed by *Tuition fees up to date* (7%). AUC is lower (0.8698) because leaf-level probabilities are coarse, which limits ranking quality. |
| **KNN** | Accuracy (0.7164) looks reasonable, but F1 (0.6941) and MCC (0.5275) reveal a bias towards the majority class. Graduate recall is the highest of any model (0.941), but Enrolled recall is the lowest (0.264): 89 of 159 Enrolled students were predicted as Graduate. After one-hot encoding the feature space has 132 dimensions, and many of them are sparse indicators. This dilutes distance-based neighbourhoods. |
| **Naive Bayes** | Lowest overall (Accuracy 0.6893, MCC 0.4879). The conditional-independence assumption is clearly violated: 1st- and 2nd-semester counterparts are strongly correlated (approved units r = 0.90, enrolled units r = 0.94, grade r = 0.84). As a result the model double-counts the same evidence. A large `var_smoothing` (0.1) was needed to stabilise the Gaussian estimates on binary one-hot columns. It is less majority-biased than KNN (Enrolled recall 0.352), but it has the weakest Graduate recall (0.842). |
| **Support Vector Machine** | A very close second (Accuracy 0.7672, AUC 0.8982, MCC 0.6141). It had the best cross-validation F1 (0.7736) but generalised slightly below Logistic Regression on the hold-out set. The non-linear RBF kernel adds no measurable gain on this data, at a much higher cost: grid search was the slowest of the five models, and the final model keeps 1,791 support vectors (about 51% of the training rows). |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It leads on every metric, is fast to train and to serve, and its coefficients are directly interpretable. The SVM is effectively tied but more expensive. All five models find **Enrolled** the hardest class (recall ≤ 0.41): it is the minority class (17.9%) and sits between Dropout and Graduate in feature space. Improving it is the main opportunity, for example with cost-sensitive thresholds or resampling. |

---

## Streamlit app

Features (as required by the assignment):

1. **Dataset upload (CSV).** Upload test data, or switch on the bundled `test_data.csv`. A
   download button provides the sample file. The upload is validated for missing columns,
   non-numeric values and unknown target labels. Column order does not matter.
2. **Model selection dropdown.** Choose any of the five classifiers. Its tuned
   hyper-parameters are shown in the sidebar.
3. **Evaluation metrics.** Accuracy, AUC, Precision, Recall, F1 and MCC are computed live on
   the uploaded data.
4. **Confusion matrix and classification report.** Counts or row-normalised view, plus a
   per-class precision/recall/F1 table.

Additional views:

- **Compare all models:** all five models scored on the uploaded file, with the best
  value per metric highlighted and a bar chart for any chosen metric.
- **Predictions:** per-row actual vs. predicted label with class probabilities. It can be
  filtered to misclassified rows and downloaded as CSV.

The app loads the saved pipelines from `model/` and never retrains on the uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py   # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

The saved models were serialised with scikit-learn 1.7.2. `requirements.txt` pins this
version so the `.joblib` files load identically on Streamlit Community Cloud. Use Python
3.11–3.13 when deploying.
