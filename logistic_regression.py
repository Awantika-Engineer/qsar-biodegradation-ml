
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load training and testing datasets
train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("test_data.csv")

# 2. Separate features and target
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# 3. Create pipeline (scaling + classifier)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# 4. Train on training data only
model.fit(X_train, y_train)

# 5. Predict test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Calculate evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# 7. Display results
print("\nLogistic Regression Evaluation")
print("-" * 40)

for name, value in metrics.items():
    print(f"{name:<12}: {value:.4f}")

# 8. Save trained pipeline
joblib.dump(model, "model/logistic_regression.joblib")

print("\nModel saved successfully!")