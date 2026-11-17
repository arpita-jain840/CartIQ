import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "laptops_cleaned.csv"

# Graph output directory
OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("\n========== CARTIQ EDA REPORT ==========")

# 1. Basic information
print("\n1. Dataset Information:")
print(df.info())

# 2. Price analysis
print("\n2. Price Statistics:")
print(df["price_eur"].describe())

# 3. Brand analysis
print("\n3. Number of Laptops by Brand:")
print(df["company"].value_counts())

# 4. RAM analysis
print("\n4. RAM Distribution:")
print(df["ram_gb"].value_counts().sort_index())

# 5. Laptop type analysis
print("\n5. Laptop Types:")
print(df["typename"].value_counts())

# 6. Average price by brand
print("\n6. Average Price by Brand:")
print(
    df.groupby("company")["price_eur"]
    .mean()
    .sort_values(ascending=False)
)

# --------------------------------
# VISUALIZATIONS
# --------------------------------

# Graph 1: Price Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["price_eur"], bins=30, edgecolor="black")
plt.title("Laptop Price Distribution")
plt.xlabel("Price (Euro)")
plt.ylabel("Number of Laptops")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_distribution.png")
plt.show()

# Graph 2: Average Price by Brand
brand_prices = (
    df.groupby("company")["price_eur"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))
brand_prices.plot(kind="bar", edgecolor="black")
plt.title("Average Laptop Price by Brand")
plt.xlabel("Brand")
plt.ylabel("Average Price (Euro)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "brand_price_comparison.png")
plt.show()

# Graph 3: RAM vs Price
ram_prices = df.groupby("ram_gb")["price_eur"].mean()

plt.figure(figsize=(8, 5))
ram_prices.plot(kind="bar", edgecolor="black")
plt.title("Average Laptop Price by RAM")
plt.xlabel("RAM (GB)")
plt.ylabel("Average Price (Euro)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ram_vs_price.png")
plt.show()

# Graph 4: Laptop Type Distribution
plt.figure(figsize=(8, 5))
df["typename"].value_counts().plot(
    kind="bar",
    edgecolor="black"
)
plt.title("Laptop Type Distribution")
plt.xlabel("Laptop Type")
plt.ylabel("Number of Laptops")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "laptop_type_distribution.png")
plt.show()

# Graph 5: Storage vs Price
storage_prices = df.groupby("storage_gb")["price_eur"].mean()

plt.figure(figsize=(9, 5))
storage_prices.plot(kind="bar", edgecolor="black")
plt.title("Average Laptop Price by Storage")
plt.xlabel("Storage (GB)")
plt.ylabel("Average Price (Euro)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "storage_vs_price.png")
plt.show()

print("\nEDA graphs saved successfully!")
print("Location:", OUTPUT_DIR)

print("\n7. Average Price by CPU Brand:")
print(df.groupby("cpu_brand")["price_eur"].mean().sort_values(ascending=False))

print("\n8. Average Price by GPU Brand:")
print(df.groupby("gpu_brand")["price_eur"].mean().sort_values(ascending=False))

print("\n9. Average Price by Laptop Type:")
print(df.groupby("typename")["price_eur"].mean().sort_values(ascending=False))

correlation = df[
    ["price_eur", "ram_gb", "storage_gb",
     "weight_kg", "inches"]
].corr()

print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("CartIQ - Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.show()



# Average Laptop Price by CPU Brand

cpu_price = df.groupby("cpu_brand")["price_eur"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
cpu_price.plot(kind="bar", edgecolor="black")

plt.title("Average Laptop Price by CPU Brand")
plt.xlabel("CPU Brand")
plt.ylabel("Average Price (Euro)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Average Laptop Price by GPU Brand

gpu_price = df.groupby("gpu_brand")["price_eur"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
gpu_price.plot(kind="bar", edgecolor="black")

plt.title("Average Laptop Price by GPU Brand")
plt.xlabel("GPU Brand")
plt.ylabel("Average Price (Euro)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("\n========== EDA COMPLETED ==========")
