import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "laptops_cleaned.csv"

# Output path
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "laptops_features.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== CARTIQ FEATURE ENGINEERING ==========")

# 1. Screen Resolution - Total Pixels
df["resolution_pixels"] = (
    df["resolution_width"] * df["resolution_height"]
)

# 2. Screen Aspect Ratio
df["aspect_ratio"] = (
    df["resolution_width"] / df["resolution_height"]
).round(2)

# 3. Extract Storage Technology
def get_storage_type(memory):

    memory = str(memory).lower()

    has_ssd = "ssd" in memory
    has_hdd = "hdd" in memory
    has_flash = "flash" in memory

    if has_ssd and has_hdd:
        return "Hybrid"

    elif has_ssd:
        return "SSD"

    elif has_hdd:
        return "HDD"

    elif has_flash:
        return "Flash Storage"

    else:
        return "Other"


df["storage_type"] = df["memory"].apply(get_storage_type)


# 4. Display Feature Engineering Results

print("\n1. New Features Created:")
print([
    "resolution_pixels",
    "aspect_ratio",
    "storage_type"
])

print("\n2. Sample Feature Values:")
print(
    df[[
        "resolution_width",
        "resolution_height",
        "resolution_pixels",
        "aspect_ratio",
        "memory",
        "storage_type"
    ]].head(10)
)

print("\n3. Storage Technology Distribution:")
print(df["storage_type"].value_counts())

print("\n4. Updated Dataset Shape:")
print(df.shape)

# 5. Save Feature Engineered Dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature Engineered Dataset Saved!")
print("Location:", OUTPUT_PATH)

print("\n========== FEATURE ENGINEERING COMPLETED ==========")
