import os
import numpy as np
import pandas as pd
from cnn_image_model import load_pretrained_cnn, get_image_embedding

CSV_PATH = "data/food-ingredients-and-recipe-dataset-with-images/versions/1/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv"
IMG_DIR = "data/food-ingredients-and-recipe-dataset-with-images/Food Images/Food Images"
OUTPUT_PATH = "model3/image_embeddings.npy"

def extract_all_embeddings(csv_path, img_dir, output_path="image_embeddings.npy"):
    df = pd.read_csv(csv_path)
    model = load_pretrained_cnn()
    
    embeddings = []
    bad_images = []
    good_images = 0

    for idx, row in df.iterrows():
        img_name = row["Image_Name"] + ".jpg"
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            bad_images.append(img_name)
            embeddings.append(np.zeros(2048))
            continue

        emb = get_image_embedding(model, img_path)
        embeddings.append(emb.flatten())
        good_images += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} images...")

    np.save(output_path, np.stack(embeddings))
    print(f"\nCompleted! Saved embeddings to {output_path}")
    print(f"Good images: {good_images}/{len(df)}")
    print(f"Bad/missing images: {len(bad_images)}/{len(df)}")

if __name__ == "__main__":
    extract_all_embeddings(
        csv_path=CSV_PATH,
        img_dir=IMG_DIR,
        output_path=OUTPUT_PATH
    )