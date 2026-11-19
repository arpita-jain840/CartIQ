
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "laptops_features.csv"

# Model directory
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load feature engineered dataset
df = pd.read_csv(DATA_PATH)

print("\n========== CARTIQ ML PREPROCESSING ==========")

# 1. Select Numerical Features
numerical_features = [
    "inches",
    "ram_gb",
    "storage_gb",
    "weight_kg",
    "resolution_pixels",
    "aspect_ratio",
    "is_touchscreen"
]

# 2. Select Categorical Features
categorical_features = [
    "company",
    "typename",
    "opsys",
    "cpu_brand",
    "gpu_brand",
    "storage_type"
]

# 3. Prepare Input Data
X = df[numerical_features + categorical_features]

print("\n1. Input Dataset Shape:")
print(X.shape)

# 4. Numerical Preprocessing
numerical_transformer = StandardScaler()

# 5. Categorical Preprocessing
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

# 6. Combine Both Transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 7. Create ML Pipeline
preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ]
)

# 8. Transform Dataset
X_processed = preprocessing_pipeline.fit_transform(X)

print("\n2. Original Feature Count:")
print(X.shape[1])

print("\n3. Processed Feature Shape:")
print(X_processed.shape)

print("\n4. Processed Feature Names:")
feature_names = (
    preprocessing_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

print(feature_names)

# 9. Save Preprocessing Pipeline
joblib.dump(
    preprocessing_pipeline,
    MODEL_DIR / "laptop_preprocessor.joblib"
)

print("\nPreprocessing Pipeline Saved Successfully!")

print("\n========== PREPROCESSING COMPLETED ==========")
