
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "laptops_cleaned.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== CARTIQ FEATURE SELECTION ==========")

# 1. Numerical Features
numerical_features = [
    "inches",
    "ram_gb",
    "storage_gb",
    "weight_kg",
    "resolution_width",
    "resolution_height",
    "is_touchscreen"
]

# 2. Categorical Features
categorical_features = [
    "company",
    "typename",
    "opsys",
    "cpu_brand",
    "gpu_brand"
]

# 3. Detailed Specifications
detailed_features = [
    "cpu",
    "gpu",
    "memory",
    "screenresolution"
]

# 4. Display Information
display_features = [
    "laptop_id",
    "product",
    "price_eur"
]

print("\n1. Numerical Features:")
print(numerical_features)

print("\n2. Categorical Features:")
print(categorical_features)

print("\n3. Detailed Specifications:")
print(detailed_features)

print("\n4. Display Information:")
print(display_features)

# 5. Unique values in categorical features
print("\n5. Unique Categories:")

for col in categorical_features:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# 6. Numerical feature statistics
print("\n6. Numerical Feature Statistics:")
print(df[numerical_features].describe())

print("\n========== FEATURE SELECTION COMPLETED ==========")
