import pandas as pd
from pathlib import Path

# CartIQ main directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "raw" / "laptop_price.csv"

# Check dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Read CSV
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\n========== CARTIQ DATASET REPORT ==========")

# Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# First five records
print("\n2. First 5 Records:")
print(df.head())

# All columns
print("\n3. Column Names:")
print(df.columns.tolist())

# Data types
print("\n4. Data Types:")
print(df.dtypes)

# Missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Duplicate records
print("\n6. Duplicate Rows:")
print(df.duplicated().sum())

# Statistical analysis
print("\n7. Statistical Summary:")
print(df.describe(include="all").T)

print("\n========== REPORT COMPLETED ==========")
