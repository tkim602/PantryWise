import os
import numpy as np
import pandas as pd
from cnn_image_model import load_pretrained_cnn, get_image_embedding

CSV_PATH = "data/food-ingredients-and-recipe-dataset-with-images/versions/1/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv"
IMG_DIR = "data/food-ingredients-and-recipe-dataset-with-images/Food Images/Food Images"

print("Loading model...")
model = load_pretrained_cnn()

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print(f"Testing extraction on first 100 images...\n")

embeddings = []
good_count = 0
bad_count = 0

for idx in range(min(100, len(df))):
    img_name = df.iloc[idx]["Image_Name"] + ".jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    
    if not os.path.exists(img_path):
        embeddings.append(np.zeros(2048))
        bad_count += 1
        if idx < 5:
            print(f"✗ {idx}: {img_name} NOT FOUND")
    else:
        emb = get_image_embedding(model, img_path)
        embeddings.append(emb.flatten())
        good_count += 1
        if idx < 5:
            non_zero = (emb.flatten() != 0).sum()
            print(f"✓ {idx}: {img_name} ({non_zero} non-zero values)")

print(f"\n{'='*60}")
print(f"Results:")
print(f"  Good images: {good_count}/100")
print(f"  Bad images: {bad_count}/100")
print(f"  Sample embedding shape: {embeddings[0].shape}")
print(f"  Sample non-zero values: {(embeddings[good_count-1] != 0).sum()}")
print(f"{'='*60}")
