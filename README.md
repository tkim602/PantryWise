# 4641Fall25Group40

This is our CS 4641 Fall 2025 Group 40 project repository.

Our website is located [here](https://github.gatech.edu/pages/eham9/4641Fall25Group40website/)

## File Structure

/data/: Contains all data files in the project

/data/food-ingredients-and-recipe-dataset-with-images/: Main dataset directory downloaded from Kaggle

/data/food-ingredients-and-recipe-dataset-with-images/Food Images/: Directory containing all food images for the recipes

/data/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.csv: Original Kaggle CSV

/data/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv: CSV with normalized ingredient text from our preprocessing pipline

/clean_data.py: Script to normalize the original Kaggle dataset

/download_data.py: Script to download the dataset

/model1/: Contains our implementation of our rule-based weighted score model

/model1/k_means.py: Main script for clustering, feature extraction, pantry coverage, and recipe recommendation

/model1/clustering_evaluation.py: Script for analyzing our k-means algorithm with Elbow Method, Silhouette Score, and Davies-Bouldin Index.

/model2_lightGBM_v2/: Updated lightGBM model directory

/model2_lightGBM_v2/pipeline_classification.py: Main pipline (classification for taste)

/model2_lightGBM_v2/pipeline.py: Base pipeliness (regression for taste)

/model2_lightGBM_v2/models/: Contains utility files

/model2_lightGBM_v2/preprocess_foodcom.py: Preprocessing utilities

/model2_lightGBM_v2/data/: Contains dataset Food.com data

/model2_lightGBM_v2/artifacts/models: Contains saved lightGBM models

/model2_lightGBM_v2/plots_classification/: Contains evaluation plots

/model3/: Stores the CNN-generated image embeddings used for visual similarity recommendations.

/model3/image_embeddings.npy: Precomputed 2048-dimensional image feature vectors (one per recipe image).

/model3/cnn_feature_extraction.py: Loads CNN model and extracts features

/model3/cnn_image_model.py: Defines CNN architecture and related utilities

/model3/image_recommender.py: Implements image-based recommendation using CNN embeddings

/model3/test_extraction.py: Test script to verity CNN works and finds expected embeddings

/model3/test_image_recommender.py: Testing script for image-based recommender

/model3/evaluate_cnn_comprehensive.py: Evaluats HR, MRR, P@k as well as other metrics

/model3/visualize_metrics.py: Generates visualizations for metrics

/model3/evaluation_plots/: Directory containing saved visualizations

## Setup

```
pip install kagglehub
python download_data.py
```

This should print the path of the download. Move the "food-ingredients-and-recipe-dataset-with-images" folder into data. Then remove the "version/1" folders if applicable. Inside the "food-ingredients-and-recipe-dataset-with-images" folder should be the "Food Images" directory and "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

## Normalize ingredient text

After downloading the dataset, you can generate a CSV with normalized ingredient text by running (make sure to make a venv first and pip install pandas)

```
python3 clean_data.py
```

To perform the tagging step, run

```
preprocess_tags.py
```

## Run model 3: CNN-based image recommender

```
python model3/cnn_feature_extraction.py
python model3/test_image_recommender.py
```

This first uses the CNN to process all recipe images and save embeddings to model3/image_embeddings.npy. Then, it loads image_embeddings.npy and the normalized CSV to return the top visually similar recipes for a given recipe_id
