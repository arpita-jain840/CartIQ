
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load laptop catalog
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(MODEL_DIR / "laptop_catalog.csv")


def recommend_laptops(
    budget,
    min_ram,
    preferred_brand=None,
    top_n=5
):

    # 1. Filter by budget
    results = df[df["price_eur"] <= budget].copy()

    # 2. Filter by minimum RAM
    results = results[
        results["ram_gb"] >= min_ram
    ]

    # 3. Apply brand preference if provided
    if preferred_brand:
        brand_results = results[
            results["company"].str.lower()
            == preferred_brand.lower()
        ]

        # If preferred brand has no matches,
        # keep other brands within budget and RAM
        if not brand_results.empty:
            results = brand_results

    # 4. Sort by RAM and price
    results = results.sort_values(
        by=["ram_gb", "price_eur"],
        ascending=[False, True]
    )

    # 5. Display top recommendations
    results = results.head(top_n)

    if results.empty:
        print("\nNo matching laptops found.")
        return

    print("\n====== CARTIQ PERSONALIZED RESULTS ======")

    for i, (_, laptop) in enumerate(
        results.iterrows(), start=1
    ):

        print(f"\nRecommendation {i}")
        print("Brand:", laptop["company"])
        print("Product:", laptop["product"])
        print("RAM:", laptop["ram_gb"], "GB")
        print("Storage:", laptop["storage_gb"], "GB")
        print("Price:", laptop["price_eur"], "EUR")


# Test the function
recommend_laptops(
    budget=1000,
    min_ram=16,
    preferred_brand="Lenovo",
    top_n=5
)
