
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset with actual feature names
df = pd.read_csv("data/biodeg_with_headers.csv")

# 2. Convert categorical target into numerical labels
# NRB = 0 (Not Readily Biodegradable)
# RB  = 1 (Readily Biodegradable)
df["target"] = df["target"].map({
    "NRB": 0,
    "RB": 1
})

# 3. Validate target encoding
assert df["target"].notna().all(), "Invalid target values found"

# 4. Separate input features and target
X = df.drop(columns=["target"])
y = df["target"].astype(int)

# 5. Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Save the training data
train_df = X_train.copy()
train_df["target"] = y_train

train_df.to_csv("data/train_data.csv", index=False)

# 7. Save test data for Streamlit evaluation
test_df = X_test.copy()
test_df["target"] = y_test

test_df.to_csv("test_data.csv", index=False)

# 8. Print verification results
print("Total instances:", len(df))
print("Training instances:", len(X_train))
print("Testing instances:", len(X_test))
print("Number of input features:", X.shape[1])

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nData preparation completed successfully!")
