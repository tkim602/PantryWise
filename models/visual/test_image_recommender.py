from image_recommender import ImageRecommender

recommender = ImageRecommender(
    embeddings_path="model3/image_embeddings.npy",
    csv_path="data/food-ingredients-and-recipe-dataset-with-images/versions/1/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv"
)

results = recommender.recommend(recipe_id=50, top_k=5)

print(results)
