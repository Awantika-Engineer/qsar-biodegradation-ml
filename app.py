
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# 1. Streamlit page configuration
st.set_page_config(
    page_title="QSAR Biodegradation | ML Classification",
    page_icon="🧪",
    layout="wide"
)

st.title("QSAR Biodegradation Classification")
st.caption("BITS Pilani | M.Tech AIML | Machine Learning Assignment 2")

st.write(
    "Evaluate five machine learning classifiers using "
    "molecular descriptors from the QSAR Biodegradation dataset."
)

# 2. Available trained models
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

# 3. Upload test dataset
st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV containing 41 features and the target column",
    type=["csv"]
)

# 4. Model selection
st.subheader("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a trained model",
    list(MODEL_FILES.keys())
)

# 5. Load selected trained model
model_path = Path(MODEL_FILES[selected_model])

if not model_path.is_file():
    st.error(f"Model file not found: {model_path}")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(model_path))

st.success(f"{selected_model} loaded successfully!")

# 6. Display uploaded dataset
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

        st.subheader("3. Uploaded Dataset Preview")

        # Expandable dataset preview
        with st.expander("View Uploaded Dataset Preview", expanded=False):
            st.dataframe(
                test_df.head(10),
                use_container_width=True
            )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(test_df))
        col2.metric("Input Features", len(test_df.columns) - 1)
        col3.metric("Target Column", "Present" if "target" in test_df.columns else "Missing")

    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, pd.errors.EmptyDataError) as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()

else:
    st.info("Upload test_data.csv to begin evaluation.")


# 7. Evaluate the selected model using uploaded test data
if uploaded_file is not None:

    st.subheader("4. Model Evaluation Results")

    # Validate that target column exists
    if "target" not in test_df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    # Separate features and target
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]


    # Validate target labels
    if y_test.isna().any():
        st.error("Target column contains missing values.")
        st.stop()

    if not y_test.isin([0, 1]).all():
        st.error(
            "Invalid target labels. "
            "Use 0 for NRB and 1 for RB."
        )
        st.stop()

    # Convert target to integer
    y_test = y_test.astype(int)

    # AUC requires both positive and negative classes
    if y_test.nunique() < 2:
        st.warning(
            "Only one target class was found. "
            "AUC cannot be calculated for this dataset."
        )


    # Verify required features are present
    expected_features = list(model.feature_names_in_)

    missing_features = [
        col for col in expected_features
        if col not in X_test.columns
    ]

    extra_features = [
        col for col in X_test.columns
        if col not in expected_features
    ]

    if missing_features or extra_features:
        st.error(
            f"Invalid feature columns. "
            f"Missing: {missing_features}, "
            f"Unexpected: {extra_features}"
        )
        st.stop()

    # Preserve the feature order used during training
    X_test = X_test[expected_features]


    # Validate numerical feature values
    try:
        X_test = X_test.apply(
            pd.to_numeric,
            errors="raise"
        )

    except (ValueError, TypeError):
        st.error(
            "Invalid dataset: All 41 input features must "
            "contain numerical values."
        )
        st.stop()

    # Check for missing values
    if X_test.isna().any().any():
        st.error(
            "Invalid dataset: Missing feature values detected."
        )
        st.stop()

    # Check for positive or negative infinity
    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        st.error(
            "Invalid dataset: Features contain infinite "
            "or non-finite numerical values."
        )
        st.stop()


    # Generate predictions without retraining
    y_pred = model.predict(X_test)

    positive_index = list(model.classes_).index(1)
    y_prob = model.predict_proba(X_test)[:, positive_index]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = (
        roc_auc_score(y_test, y_prob)
        if y_test.nunique() == 2
        else None
    )

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Display metrics in two rows
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # 8. Confusion Matrix
    st.subheader("5. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NRB (0)", "RB (1)"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title(f"{selected_model} - Confusion Matrix")

    st.pyplot(fig, width="content")
    plt.close(fig)

    st.caption(
    "Rows represent actual classes and columns represent predicted classes. "
    "Diagonal cells indicate correct predictions, while off-diagonal cells "
    "indicate misclassifications. Darker shades represent higher sample counts."
    )

    # 9. Classification Report
    st.subheader("6. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["NRB", "RB"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    # Accuracy is already displayed separately in evaluation metrics.
    # Remove the duplicated accuracy row to avoid misleading columns.
    report_df = report_df.drop(index="accuracy", errors="ignore")

    st.dataframe(
        report_df.style.format({
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1-score": "{:.4f}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    st.success(
        f"{selected_model} evaluation completed successfully!"
    )
