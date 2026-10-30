
import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB
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

# 3. Initialize Gaussian Naive Bayes
model = GaussianNB()

# 4. Train on training data only
model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Calculate evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

# 7. Display results
print("\nGaussian Naive Bayes Evaluation")
print("-" * 40)

for name, value in metrics.items():
    print(f"{name:<12}: {value:.4f}")

# 8. Save trained model
joblib.dump(model, "model/naive_bayes.joblib")

print("\nModel saved successfully!")