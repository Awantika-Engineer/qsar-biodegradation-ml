
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load the same test dataset used for all five models
test_df = pd.read_csv("test_data.csv")

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# 2. Define saved model locations
model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

results = []


# 3. Evaluate all saved models
for model_name, model_path in model_files.items():

    model = joblib.load(model_path)

    # Validate target class labels
    assert list(model.classes_) == [0, 1], (
        f"Unexpected class labels for {model_name}"
    )

    # Validate feature names and their order
    assert list(X_test.columns) == list(model.feature_names_in_), (
        f"Feature mismatch for {model_name}"
    )

    print(f"{model_name}: Validation passed")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get probabilities for the positive class (RB = 1)
    positive_index = list(model.classes_).index(1)
    y_prob = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)


# 4. Create comparison table
comparison_df = pd.DataFrame(results)

print("\nML Model Comparison")
print("-" * 90)
print(comparison_df.to_string(index=False, float_format="%.4f"))

# 5. Save comparison results
Path("results").mkdir(exist_ok=True)

comparison_df.to_csv(
    "results/model_comparison.csv",
    index=False,
    float_format="%.12f"
)

print("\nComparison saved to results/model_comparison.csv")

# 6. Display best model by each evaluation metric
print("\nBest Model by Metric")
print("-" * 40)

for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    best_row = comparison_df.loc[comparison_df[metric].idxmax()]
    print(f"{metric:<12}: {best_row['Model']} ({best_row[metric]:.4f})")