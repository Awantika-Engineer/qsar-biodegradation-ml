# QSAR Biodegradation Classification

## 1. Problem Statement

The objective of this project is to classify chemical compounds as Readily Biodegradable (RB) or Not Readily Biodegradable (NRB) using molecular descriptors.

Five supervised machine learning classification algorithms are implemented and evaluated using six performance metrics.

An interactive Streamlit application allows users to upload a labeled test dataset, select a trained classifier, and view evaluation results without retraining the model.

## 2. Dataset Description

**Dataset:** QSAR Biodegradation  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/254/qsar+biodegradation

| Property | Description |
|---|---|
| Total instances | 1,055 |
| Input features | 41 |
| Classification type | Binary |
| Target variable | Biodegradability |
| Positive class | RB (1) |
| Negative class | NRB (0) |
| RB instances | 356 |
| NRB instances | 699 |
| Missing values | None |
| Training instances | 844 (80%) |
| Testing instances | 211 (20%) |

The dataset contains numerical molecular descriptors representing chemical characteristics.

The original dataset is semicolon-delimited and does not contain a header row. Feature names were assigned according to the dataset documentation.

The target variable was encoded as NRB = 0 and RB = 1. A stratified train-test split was performed using `random_state=42` to preserve the class distribution.

## 3. GitHub Repository

**Repository:** https://github.com/Awantika-Engineer/qsar-biodegradation-ml

The repository contains the Python source code, saved trained models, dataset files, evaluation results, requirements file, and Streamlit application.

## 4. Machine Learning Models

The following five classification algorithms were implemented using Scikit-learn:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

StandardScaler was applied using pipelines for Logistic Regression, KNN, and SVM.

For SVM, an RBF kernel was used with sigmoid probability calibration through CalibratedClassifierCV (5-fold cross-validation, ensemble=False) to obtain class probabilities for AUC evaluation.

All models were trained using the same training dataset and evaluated on the same held-out test dataset.

## 5. Model Evaluation Results

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8626 | 0.9136 | 0.8000 | 0.7887 | 0.7943 | 0.6912 |
| Decision Tree | 0.8152 | 0.7844 | 0.7424 | 0.6901 | 0.7153 | 0.5796 |
| KNN | 0.8531 | 0.8973 | 0.7703 | 0.8028 | 0.7862 | 0.6747 |
| Gaussian Naive Bayes | 0.6967 | 0.8650 | 0.5276 | 0.9437 | 0.6768 | 0.4972 |
| SVM | 0.8531 | 0.9189 | 0.7941 | 0.7606 | 0.7770 | 0.6679 |

All metrics were calculated using the test dataset containing 211 observations.

### Evaluation Metrics and Terminology

**Class Labels**
- **RB (1):** Readily Biodegradable – Positive class.
- **NRB (0):** Not Readily Biodegradable – Negative class.

**Evaluation Metrics**
- **Accuracy:** Proportion of correctly classified samples.
- **AUC (Area Under the ROC Curve):** Measures the model's ability to distinguish between RB and NRB across different classification thresholds.
- **Precision:** Proportion of predicted RB samples that are actually RB.
- **Recall:** Proportion of actual RB samples correctly identified.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **MCC (Matthews Correlation Coefficient):** Measures classification agreement using TP, TN, FP and FN. Values range from −1 to +1.

**Classification Report Terms**
- **Macro Avg:** Unweighted average of a metric across both classes, giving each class equal importance.
- **Weighted Avg:** Average of a metric across both classes, weighted by the number of actual samples in each class.
- **Support:** Number of actual samples belonging to each class.


## 6. Model Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | Achieved the highest accuracy (86.26%), precision (80%), F1 score (0.7943), and MCC (0.6912), demonstrating strong overall classification performance. |
| Decision Tree | Achieved 81.52% accuracy, but lower AUC and MCC than Logistic Regression. Further training-set analysis would be needed to determine whether overfitting contributed to these results. |
| KNN | Achieved 85.31% accuracy and 80.28% recall. Its performance was close to Logistic Regression, with slightly higher recall but lower precision, F1 and MCC. |
| Gaussian Naive Bayes | Achieved the highest recall (94.37%) but comparatively low precision (52.76%), suggesting that it identifies most RB compounds while producing more false-positive classifications. |
| SVM | The SVM model with an RBF kernel and sigmoid probability calibration achieved the highest AUC (0.9189), indicating strong discrimination between RB and NRB classes. It achieved 85.31% accuracy, with F1 (0.7770) and MCC (0.6679) slightly below Logistic Regression. |

### Overall Winning Model: Logistic Regression

Logistic Regression was selected as the best overall model for this experiment.

It achieved the highest Accuracy, Precision, F1 Score, and MCC among the five classifiers.

Although SVM achieved a slightly higher AUC and Gaussian Naive Bayes achieved higher Recall, Logistic Regression provided the strongest overall balance of classification metrics on the held-out test dataset.

## 7. Streamlit Application

**Live Application:** [Add URL after deployment]

The Streamlit application supports:

- CSV upload containing 41 input features and the target column.
- Selection of any of the five trained classifiers.
- Evaluation using Accuracy, AUC, Precision, Recall, F1, and MCC.
- Confusion matrix visualization.
- Classification report.
- CSV validation for missing, invalid, or unexpected values.

The application loads saved `.joblib` models and does not retrain models when evaluating uploaded test data.

## 8. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/Awantika-Engineer/qsar-biodegradation-ml.git
cd qsar-biodegradation-ml
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Upload `test_data.csv` to evaluate the saved trained models.

## 9. Project Files

| File / Directory | Purpose |
|---|---|
| `check_dataset.py` | Dataset inspection and column naming |
| `prepare_data.py` | Target encoding and train-test split |
| `logistic_regression.py` | Logistic Regression training |
| `decision_tree.py` | Decision Tree training |
| `knn.py` | KNN training |
| `naive_bayes.py` | Gaussian Naive Bayes training |
| `svm.py` | SVM training |
| `compare_models.py` | Evaluation and model comparison |
| `app.py` | Streamlit application |
| `test_data.csv` | Held-out test dataset |
| `model/` | Saved trained classification models |
| `data/` | Dataset files |
| `results/` | Evaluation comparison results |
| `requirements.txt` | Python dependencies |

## 10. References

UCI Machine Learning Repository – QSAR Biodegradation  
https://archive.ics.uci.edu/dataset/254/qsar+biodegradation

Scikit-learn Documentation  
https://scikit-learn.org/stable/

Streamlit Documentation  
https://docs.streamlit.io/
