
import pandas as pd

# 1. Original feature names from UCI (in CSV column order)
feature_names = [
    "SpMax_L",
    "J_Dz(e)",
    "nHM",
    "F01[N-N]",
    "F04[C-N]",
    "NssssC",
    "nCb-",
    "C%",
    "nCp",
    "nO",
    "F03[C-N]",
    "SdssC",
    "HyWi_B(m)",
    "LOC",
    "SM6_L",
    "F03[C-O]",
    "Me",
    "Mi",
    "nN-N",
    "nArNO2",
    "nCRX3",
    "SpPosA_B(p)",
    "nCIR",
    "B01[C-Br]",
    "B03[C-Cl]",
    "N-073",
    "SpMax_A",
    "Psi_i_1d",
    "B04[C-Br]",
    "SdO",
    "TI2_L",
    "nCrt",
    "C-026",
    "F02[C-N]",
    "nHDon",
    "SpMax_B(m)",
    "Psi_i_A",
    "nN",
    "SM6_B(m)",
    "nArCOOR",
    "nX"
]

# 2. Read original dataset (without headers)
df = pd.read_csv(
    "data/biodeg.csv",
    sep=";",
    header=None
)

# 3. Validate dataset structure
assert len(feature_names) == 41, "Expected 41 features"
assert df.shape == (1055, 42), "Unexpected dataset shape"

# 4. Assign actual column names
df.columns = feature_names + ["target"]

# 5. Verify dataset
print("Dataset shape:", df.shape)
print("Number of features:", len(feature_names))
print("Missing values:", df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["target"].value_counts())

# 6. Save a new CSV with headers
df.to_csv("data/biodeg_with_headers.csv", index=False)

print("\nCSV with original feature names saved successfully!")
