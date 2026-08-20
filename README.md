# Pantry-Aware Recipe Recommender

A Georgia Tech CS 4641 machine learning project exploring pantry-aware recipe recommendation across ingredient availability, taste, nutrition, sustainability, and visual similarity.

**Approaches:** TF-IDF + K-Means ranking, LightGBM models, and ResNet50 image embeddings.

[View the final report](https://raw.githack.com/tkim602/PantryWise/main/report/index.html) · [HTML source](report/index.html)

## Repository

- `models/kmeans/` — pantry-aware K-Means / TF-IDF ranking
- `models/lightgbm/v2/` — final LightGBM taste, nutrition, and sustainability pipeline
- `models/lightgbm/v1/` — earlier LightGBM experiments
- `models/visual/` — ResNet50-based visual retrieval
- `clean_data.py`, `preprocess_tags.py` — preprocessing utilities
- `report/` — full final project report

> Course group project. See the final report for team contributions and full evaluation details.
