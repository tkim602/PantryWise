# Pantry-Aware Recipe Recommender

A Georgia Tech CS 4641 machine learning project exploring pantry-aware recipe recommendation across ingredient availability, taste, nutrition, sustainability, and visual similarity.

**Approaches:** TF-IDF + K-Means ranking, LightGBM models, and ResNet50 image embeddings.

[View the final report](https://raw.githack.com/tkim602/pantry-aware-recipe-recommender/main/report/index.html) · [HTML source](report/index.html)

## Repository

- `model1/` — pantry-aware K-Means / TF-IDF ranking
- `model2_lightGBM_v2/` — LightGBM taste, nutrition, and sustainability pipeline
- `model2_lightGBM_v1/` — earlier LightGBM experiments
- `model3/` — ResNet50-based visual retrieval
- `clean_data.py`, `preprocess_tags.py` — preprocessing utilities
- `report/` — final project report

> Course group project. See the final report for team contributions and full evaluation details.
