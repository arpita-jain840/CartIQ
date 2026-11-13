import pandas as pd
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
    .str.replace("GB", "", regex=False)
    .astype(int)
)

# 5. Convert Weight into numerical KG
df["weight_kg"] = (
    df["weight"]
    .str.replace("kg", "", regex=False)
    .astype(float)
)

# 6. Extract screen resolution
def get_resolution(value):
    for part in str(value).split():
        if "x" in part:
            dimensions = part.split("x")

            if len(dimensions) == 2:
                try:
                    return int(dimensions[0]), int(dimensions[1])
                except ValueError:
                    continue

    return None, None


resolution = df["screenresolution"].apply(get_resolution)

df["resolution_width"] = resolution.apply(lambda x: x[0])
df["resolution_height"] = resolution.apply(lambda x: x[1])

# 7. Extract touchscreen information
df["is_touchscreen"] = (
    df["screenresolution"]
    .str.contains("Touchscreen", case=False, na=False)
    .astype(int)
)

# 8. Extract total storage in GB
def convert_storage(value):
    total = 0

    parts = str(value).upper().replace("+", " ").split()

    for part in parts:
        if part.endswith("GB"):
            amount = part.replace("GB", "")
            try:
                total += float(amount)
            except ValueError:
                continue

        elif part.endswith("TB"):
            amount = part.replace("TB", "")
            try:
                total += float(amount) * 1024
            except ValueError:
                continue

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

# 13. Generate cleaning report
print("\n========== CARTIQ CLEANING REPORT ==========")

print("\nDataset Shape:")
print(df.shape)

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

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Records:")
print(df.head())

print("\nProcessed Dataset Saved At:")
print(OUTPUT_PATH)

print("\n========== CLEANING COMPLETED ==========")