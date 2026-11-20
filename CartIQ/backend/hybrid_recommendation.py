
import pandas as pd
import joblib

from pathlib import Path

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# 2. Load saved files
features_matrix = joblib.load(
    MODEL_DIR / "laptop_features_matrix.joblib"
)

recommendation_model = joblib.load(
    MODEL_DIR / "laptop_recommender.joblib"
)

laptop_catalog = pd.read_csv(
    MODEL_DIR / "laptop_catalog.csv"
)

print("\n========== CARTIQ HYBRID ENGINE ==========")


def hybrid_recommend(
    laptop_index,
    budget,
    min_ram,
    preferred_brand=None,
    top_n=5
):

    # Validate selected laptop
    if laptop_index < 0 or laptop_index >= len(laptop_catalog):
        print("Invalid laptop index!")
        return

    # 3. Filter products by budget
    candidates = laptop_catalog[
        laptop_catalog["price_eur"] <= budget
    ].copy()

    # 4. Filter by minimum RAM
    candidates = candidates[
        candidates["ram_gb"] >= min_ram
    ]

    # 5. Apply preferred brand
    if preferred_brand:
        brand_matches = candidates[
            candidates["company"].str.lower()
            == preferred_brand.lower()
        ]

        # Keep other brands if no preferred-brand match exists
        if not brand_matches.empty:
            candidates = brand_matches

    if candidates.empty:
        print("\nNo matching laptops found!")
        return

    # 6. Find similarity with selected laptop
    distances, indices = recommendation_model.kneighbors(
        features_matrix[laptop_index],
        n_neighbors=len(laptop_catalog)
    )

    # 7. Create similarity ranking
    similarity_scores = {}

    for distance, index in zip(distances[0], indices[0]):
        similarity_scores[index] = (1 - distance) * 100

    # 8. Attach similarity scores to filtered products
    candidates["similarity"] = candidates.index.map(
        similarity_scores
    )

    candidates = candidates.dropna(subset=["similarity"])

    # 9. Sort by similarity
    candidates = candidates.sort_values(
        by="similarity",
        ascending=False
    )

    # Remove selected laptop itself
    candidates = candidates[
        candidates.index != laptop_index
    ]

    # 10. Display results
    print("\nSelected laptop:")
    selected = laptop_catalog.iloc[laptop_index]
    print(selected["company"], selected["product"])

    print("\n========== HYBRID RECOMMENDATIONS ==========")

    results = candidates.head(top_n)

    if results.empty:
        print("No alternative recommendations found!")
        return

    for rank, (_, laptop) in enumerate(
        results.iterrows(), start=1
    ):

        print(f"\nRecommendation {rank}")
        print("Brand:", laptop["company"])
        print("Product:", laptop["product"])
        print("RAM:", laptop["ram_gb"], "GB")
        print("Storage:", laptop["storage_gb"], "GB")
        print("Price:", laptop["price_eur"], "EUR")
        print(
            "Similarity:",
            round(laptop["similarity"], 2),
            "%"
        )


# 11. Test the hybrid engine
hybrid_recommend(
    laptop_index=0,
    budget=1000,
    min_ram=8,
    preferred_brand="Lenovo",
    top_n=5
)

print("\n========== HYBRID ENGINE COMPLETED ==========")
