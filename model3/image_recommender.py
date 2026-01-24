import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ImageRecommender:
    def __init__(self, embeddings_path, csv_path):
        self.embeddings = np.load(embeddings_path)
        self.df = pd.read_csv(csv_path)

    def recommend(self, recipe_id, top_k=5):
        target = self.embeddings[recipe_id].reshape(1, -1)
        sims = cosine_similarity(target, self.embeddings)[0]

        top_indices = sims.argsort()[::-1][1:top_k+1]
        return self.df.iloc[top_indices][["Title", "Image_Name"]]