"""
LightGBM-Based Recipe Recommender
---------------------------------
Recommends recipes using pantry coverage, TF-IDF similarity,
and sustainability scoring.
"""

import ast
import re
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Ensure NLTK lemmatizer is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()


DESCRIPTORS = [
    'unsalted','salted','unseasoned','seasoned','skinless','boneless','fresh','dried',
    'minced','chopped','sliced','grated','ground','whole','crushed','shredded','peeled',
    'halved','quartered','cubed','cut','pieces','thinly','thickly','small','large',
    'medium','organic','low fat','full fat','plain','raw','cooked','softened','melted',
    'room temperature','cold','warm','hot','extra virgin','virgin','light','dark','fine',
    'coarse','powdered','confectioners','granulated','caster','packed','optional',
    'divided','about','approx','approximately','roughly','with','without','for'
]

ALIASES = {
    'scallion': 'green onion','spring onion': 'green onion',
    'caster sugar': 'sugar','granulated sugar': 'sugar',
    'kosher salt': 'salt','sea salt': 'salt',
    'bell pepper': 'capsicum'
}

def normalize_ingredient(ing):
    ing = ing.lower()
    for d in DESCRIPTORS:
        ing = re.sub(rf"\b{re.escape(d)}\b", "", ing)
    ing = re.sub(r"\s+", " ", ing).strip()
    ing = " ".join(lemmatizer.lemmatize(w) for w in ing.split())
    return ALIASES.get(ing, ing)

def parse_ingredients(ing_str):
    if pd.isna(ing_str):
        return []
    try:
        parsed = ast.literal_eval(ing_str)
        if isinstance(parsed, list):
            return [normalize_ingredient(str(i)) for i in parsed]
    except:
        pass
    return [normalize_ingredient(w) for w in ing_str.split(",")]

def fuzzy_match(ing, pantry, threshold=85):
    return any(fuzz.token_set_ratio(ing, p) >= threshold for p in pantry)

def compute_pantry_coverage(recipe_ings, pantry):
    if not recipe_ings:
        return 0.0
    match_count = sum((ing in pantry) or fuzzy_match(ing, pantry) for ing in recipe_ings)
    return match_count / len(recipe_ings)


CARBON_FOOTPRINT_KG = {
    'beef': 60.0, 'lamb': 24.0, 'pork': 12.0, 'chicken': 6.9, 'fish': 6.1,
    'tofu': 2.0, 'lentils': 0.9, 'beans': 0.9, 'chickpeas': 0.9,
    'mushroom': 0.9, 'broccoli': 0.5, 'tomato': 0.5, 'onion': 0.4, 'garlic': 0.6,
    'rice': 2.7, 'bread': 1.1
}

DEFAULT_FOOTPRINT = 5.0

def sustainability_score(ings):
    if not ings:
        return 0.0
    score = 0.0
    for ing in ings:
        score += CARBON_FOOTPRINT_KG.get(ing, DEFAULT_FOOTPRINT)
    return score

def build_features(df, pantry_list):
    df = df.copy()
    
    # TF-IDF similarity
    ingredient_texts = df["Normalized_Ingredients"].apply(
        lambda x: " ".join(parse_ingredients(x))
    )
    vectorizer = TfidfVectorizer(max_features=4000, min_df=2, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(ingredient_texts)

    pantry_text = " ".join(pantry_list)
    pantry_vec = vectorizer.transform([pantry_text])
    df["tfidf_similarity"] = cosine_similarity(tfidf_matrix, pantry_vec).ravel()

    # Pantry coverage + sustainability
    df["pantry_coverage"] = df["Normalized_Ingredients"].apply(
        lambda x: compute_pantry_coverage(parse_ingredients(x), pantry_list)
    )

    raw_sus = df["Normalized_Ingredients"].apply(
        lambda x: sustainability_score(parse_ingredients(x))
    )
    df["sustainability_raw"] = raw_sus
    df["sustainability_score"] = 1.0 / (1.0 + df["sustainability_raw"])

    return df

def train_lightgbm(df):
    df = df.copy()
    features = ["pantry_coverage", "tfidf_similarity", "sustainability_score"]

    df["label"] = (
        0.5 * df["pantry_coverage"]
        + 0.3 * df["tfidf_similarity"]
        + 0.2 * df["sustainability_score"]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        df[features], df["label"], test_size=0.2, random_state=42
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=20,
        min_data_in_leaf=5,
        verbose=-1
    )

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=250,
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    return model, features

def recommend_lightgbm(model, df, feature_cols, top_k=10):
    df = df.copy()
    df["pred_score"] = model.predict(df[feature_cols])
    ranked = df.sort_values("pred_score", ascending=False).head(top_k)
    return ranked[["Title", "pantry_coverage", "tfidf_similarity",
                    "sustainability_score", "pred_score"]]

def main():
    data_path = Path("../data/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv")
    df = pd.read_csv(data_path)

    pantry = ["onion", "garlic", "olive oil", "salt", "rice", "tomato"]

    df = build_features(df, pantry)
    model, features = train_lightgbm(df)
    recs = recommend_lightgbm(model, df, features, top_k=10)

    print("\n=== TOP RECOMMENDATIONS ===")
    print(recs.to_string(index=False))

    return recs

if __name__ == "__main__":
    main()
    
