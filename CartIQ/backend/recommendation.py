import pandas as pd
import joblib 
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
features_matrix = joblib.load(MODEL_DIR / "laptop_features_matrix.joblib")
laptop_catalog = pd.read_csv(MODEL_DIR / "laptop_catalog.csv")
print("\n========== CARTIQ RECOMMENDATION ENGINE ==========")








print("\nTotal laptops:", features_matrix.shape[0])
print("Total features:", features_matrix.shape[1])

# 5. Create Nearest Neighbors model
recommendation_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6
)

# 6. Train model
recommendation_model.fit(features_matrix)

print("\nRecommendation Model Trained Successfully!")

# 7. Save trained model
joblib.dump(
    recommendation_model,
    MODEL_DIR / "laptop_recommender.joblib"
)

print("Recommendation Model Saved Successfully!")


# 8. Recommendation function
def recommend_similar_laptops(laptop_index, top_n=5):

    if laptop_index < 0 or laptop_index >= len(laptop_catalog):
        print("Invalid laptop index!")
        return

    # Get selected laptop
    selected_laptop = laptop_catalog.iloc[laptop_index]

    print("\nSelected Laptop:")
    print(selected_laptop["company"], selected_laptop["product"])

    # Find nearest laptops
    distances, indices = recommendation_model.kneighbors(
        features_matrix[laptop_index],
        n_neighbors=min(top_n + 1, len(laptop_catalog))
    )

    print("\n========== TOP RECOMMENDATIONS ==========")

    count = 0

    for distance, index in zip(distances[0], indices[0]):

        # Skip the selected laptop itself
        if index == laptop_index:
            continue

        similarity = (1 - distance) * 100

        laptop = laptop_catalog.iloc[index]

        count += 1

        print(f"\nRecommendation {count}")
        print("Brand:", laptop["company"])
        print("Product:", laptop["product"])
        print("RAM:", laptop["ram_gb"], "GB")
        print("Storage:", laptop["storage_gb"], "GB")
        print("Price:", laptop["price_eur"], "EUR")
        print("Similarity:", round(similarity, 2), "%")

        if count == top_n:
            break


# 9. Test recommendation system
recommend_similar_laptops(laptop_index=0, top_n=5)

print("\n========== CARTIQ RECOMMENDATION COMPLETED ==========")
