
import pandas as pd
import re
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
INPUT_PATH = BASE_DIR / "data" / "raw" / "laptop_price.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "laptops_cleaned.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH, encoding="latin1")

print("Starting CartIQ Data Preparation...")

# 1. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# 2. Remove duplicate records
df.drop_duplicates(inplace=True)

# 3. Clean text columns
text_columns = df.select_dtypes(include=["object", "str"]).columns

for col in text_columns:
    df[col] = df[col].str.strip()

# 4. Convert RAM into numerical GB
df["ram_gb"] = (
    df["ram"]
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
    .astype(float)
)

# 5. Convert weight into numerical KG
df["weight_kg"] = (
    df["weight"]
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
    .astype(float)
)

# 6. Extract screen resolution
resolution = df["screenresolution"].str.extract(
    r"(\d{3,5})x(\d{3,5})"
)

df["resolution_width"] = pd.to_numeric(
    resolution[0], errors="coerce"
)

df["resolution_height"] = pd.to_numeric(
    resolution[1], errors="coerce"
)

# 7. Extract touchscreen information
df["is_touchscreen"] = (
    df["screenresolution"]
    .str.contains("Touchscreen", case=False, na=False)
    .astype(int)
)

# 8. Extract total storage in GB
def convert_storage(value):
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(TB|GB)",
        str(value),
        re.IGNORECASE
    )

    total = 0

    for amount, unit in matches:
        amount = float(amount)

        if unit.upper() == "TB":
            total += amount * 1024
        else:
            total += amount

    return total

df["storage_gb"] = df["memory"].apply(convert_storage)

# 9. Extract CPU brand
df["cpu_brand"] = df["cpu"].str.split().str[0]

# 10. Extract GPU brand
df["gpu_brand"] = df["gpu"].str.split().str[0]

# 11. Rename price column
df.rename(
    columns={"price_euros": "price_eur"},
    inplace=True
)

# 12. Save processed dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\n========== CARTIQ CLEANING REPORT ==========")

print("\nDataset Shape:", df.shape)

print("\nNew Features:")
print([
    "ram_gb",
    "weight_kg",
    "resolution_width",
    "resolution_height",
    "is_touchscreen",
    "storage_gb",
    "cpu_brand",
    "gpu_brand"
])

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nProcessed Dataset Saved At:")
print(OUTPUT_PATH)

print("\n========== CLEANING COMPLETED ==========")
