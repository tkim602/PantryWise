import kagglehub

# Download latest version
path = kagglehub.dataset_download("pes12017000148/food-ingredients-and-recipe-dataset-with-images")

print("Path to dataset files:", path)