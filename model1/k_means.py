
from clustering_evaluation import ClusteringEvaluator
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import re
from rapidfuzz import fuzz
from nltk.stem import WordNetLemmatizer
import nltk
nltk.data.find('corpora/wordnet')


DESCRIPTORS = [
    'unsalted', 'salted', 'unseasoned', 'seasoned', 'short grain', 'long grain',
    'skinless', 'boneless', 'fresh', 'dried', 'minced', 'chopped', 'sliced', 'grated',
    'ground', 'whole', 'crushed', 'shredded', 'peeled', 'halved', 'quartered', 'cubed',
    'cut into', 'cut', 'into', 'pieces', 'thinly', 'thickly', 'small', 'large', 'medium',
    'organic', 'low fat', 'full fat', 'reduced fat', 'plain', 'raw', 'cooked', 'softened',
    'melted', 'room temperature', 'cold', 'warm', 'hot', 'extra virgin', 'virgin', 'light',
    'dark', 'fine', 'coarse', 'fine grain', 'coarse grain', 'fine-grain', 'coarse-grain',
    'extra', 'super', 'superfine', 'super-fine', 'powdered', 'confectioners', 'granulated',
    'caster', 'packed', 'loosely', 'firmly', 'drained', 'rinsed', 'to taste', 'plus more',
    'for serving', 'for garnish', 'for dusting', 'for frying', 'for greasing', 'for brushing',
    'for topping', 'for decoration', 'for drizzling', 'for sprinkling', 'for the pan',
    'as needed', 'optional', 'divided', 'about', 'approx', 'approximately', 'roughly',
    'and', 'or', 'with', 'without', 'in', 'on', 'at', 'from', 'by', 'of', 'a', 'an', 'the'
]

_lemmatizer = None
def get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is not None:
        return _lemmatizer
    if WordNetLemmatizer:
        _lemmatizer = WordNetLemmatizer()
    else:
        _lemmatizer = None
    return _lemmatizer

def normalize_ingredient(ingredient):
    """
    lowercase, remove descriptors, lemmatize, and strip whitespace.
    """
    s = ingredient.lower()
    # remove descriptors
    for desc in DESCRIPTORS:
        s = re.sub(r'\b' + re.escape(desc) + r'\b', '', s)
    # remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    # lemmatize each word
    lemmatizer = get_lemmatizer()
    if lemmatizer:
        s = ' '.join([lemmatizer.lemmatize(word) for word in s.split()])
    return s

def fuzzy_match(ing, pantry_list, threshold=85):
    """
    Return True if ing matches any pantry item above threshold using rapidfuzz's token_set_ratio.
    """
    if not fuzz:
        return False
    for p in pantry_list:
        score = fuzz.token_set_ratio(ing, p)
        if score >= threshold:
            return True
    return False

ALIASES = {
    'scallion': 'green onion',
    'spring onion': 'green onion',
    'caster sugar': 'sugar',
    'granulated sugar': 'sugar',
    'plain flour': 'all purpose flour',
    'allpurpose flour': 'all purpose flour',
    'kosher salt': 'salt',
    'sea salt': 'salt',
    'black peppercorns': 'black pepper',
    'bell pepper': 'capsicum',
}

SUSTAINABILITY = {
    # Example: positive for plants/legumes, negative for meat
    'beef': -2, 'lamb': -2, 'pork': -1, 'chicken': -0.5, 'fish': -0.5,
    'tofu': +2, 'tempeh': +2, 'lentils': +1.5, 'beans': +1.5, 'chickpeas': +1.2,
    'mushroom': +1.0, 'broccoli': +1.0, 'tomato': +0.8, 'onion': +0.7, 'garlic': +0.7,
    'rice': -0.2,  # methane footprint
}

CARBON_FOOTPRINT_KG = {
    'beef': 60.0, 'lamb': 24.0, 'pork': 12.0, 'chicken': 6.9, 'fish': 6.1,
    'tofu': 2.0, 'tempeh': 2.0, 'lentils': 0.9, 'beans': 0.9, 'chickpeas': 0.9,
    'mushroom': 0.9, 'broccoli': 0.5, 'tomato': 0.5, 'onion': 0.4, 'garlic': 0.6,
    'rice': 2.7, 'potato': 0.3, 'bread': 1.1,
}

# fallback 
DEFAULT_FOOTPRINT = 5.0 


def estimate_weight_from_name(name: str) -> float:
    """estimate weight (kg) for a single count of an ingredient based on name
    heuristic fallback used when no explicit unit is provided.
    """
    name = name.lower()
    # common whole-item approximations
    lookup = {
        'whole chicken': 1.7, 'chicken': 1.0, 'loaf': 0.5, 'apple': 0.18,
        'onion': 0.15, 'potato': 0.2, 'egg': 0.05, 'clove': 0.003, 'slice': 0.03,
    }
    for key, w in lookup.items():
        if key in name:
            return w
    # default per-item weight
    return 0.1


def convert_to_kg(quantity: float, unit: str, name: str = "") -> float:
    """quantity w/ unit -> kilograms.

    unit is empty -> use a heuristic estimate based on the ingredient name
    """
    if not unit:
        return quantity * estimate_weight_from_name(name)

    u = unit.lower()
    conversions = {
        'kg': 1.0, 'kilogram': 1.0, 'kilograms': 1.0,
        'g': 0.001, 'gram': 0.001, 'grams': 0.001,
        'lb': 0.453592, 'lbs': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
        'oz': 0.0283495, 'ounce': 0.0283495, 'ounces': 0.0283495,
        'cup': 0.24, 'cups': 0.24,
        'tbsp': 0.015, 'tablespoon': 0.015, 'tablespoons': 0.015,
        'tsp': 0.005, 'teaspoon': 0.005, 'teaspoons': 0.005,
        'ml': 0.001, 'milliliter': 0.001, 'milliliters': 0.001,
        'l': 1.0, 'liter': 1.0, 'liters': 1.0,
        'piece': 0.1, 'pieces': 0.1, 'slice': 0.03, 'slices': 0.03,
        'clove': 0.003, 'cloves': 0.003,
        'head': 0.5, 'heads': 0.5,
        'can': 0.4, 'cans': 0.4,
        'package': 0.5, 'packages': 0.5,
    }
    if u in conversions:
        return quantity * conversions[u]

    return quantity * estimate_weight_from_name(name)

def alias_map(token):
    return ALIASES.get(token, token)

def sustainability_score(ings):
    """computes quantity-weighted carbon emissions estimate (kg CO2e)
    for ings

    ret: total kg CO2e (float). Lower is better (more sustainable).
    """
    # parsed dict form
    total_emissions = 0.0
    if not ings:
        return 0.0

    # detect form: list of dicts vs list of names
    if isinstance(ings, (list, tuple)) and len(ings) > 0 and isinstance(ings[0], dict):
        for ing in ings:
            qty = ing.get('quantity', 1.0)
            unit = ing.get('unit', '') or ''
            name = ing.get('name', '')
            key = alias_map(name)
            # convert quantity -> kg
            qty_kg = convert_to_kg(qty, unit, name)
            footprint = CARBON_FOOTPRINT_KG.get(key, CARBON_FOOTPRINT_KG.get(name, DEFAULT_FOOTPRINT))
            total_emissions += qty_kg * footprint
        return float(total_emissions)

    # fallback
    try:
        return float(sum(SUSTAINABILITY.get(x, 0.0) for x in ings))
    except Exception:
        return 0.0

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def parse_ingredients(ingredients_str):
    if pd.isna(ingredients_str):
        return []
    try:
        return ast.literal_eval(ingredients_str)
    except:
        return []


def compute_recipe_metrics(df: pd.DataFrame, pantry_ingredients):
    """computes arrays of pantry coverage, complexity scores and sustainability
    emissions for all rows in `df`.

    Ret: three numpy arrays: (coverages, complexity_scores, sustainability_emissions)
    """
    coverages = []
    complexity_scores = []
    sustainability_emissions = []

    for idx, row in df.iterrows():
        recipe_ings = parse_ingredients(row['Normalized_Ingredients'])
        coverage = compute_pantry_coverage(recipe_ings, pantry_ingredients)
        complexity = recipe_complexity_score(len(recipe_ings))
        coverages.append(coverage)
        complexity_scores.append(complexity)

        parsed_col = row.get('Parsed_Ingredients', None)
        parsed_ings = None
        if isinstance(parsed_col, str) and parsed_col:
            try:
                parsed_ings = ast.literal_eval(parsed_col)
            except Exception:
                parsed_ings = None
        elif isinstance(parsed_col, (list, tuple)):
            parsed_ings = parsed_col

        if parsed_ings:
            sustainability_emissions.append(sustainability_score(parsed_ings))
        else:
            sustainability_emissions.append(sustainability_score(recipe_ings))

    return np.array(coverages), np.array(complexity_scores), np.array(sustainability_emissions)


def ingredients_to_text(ingredients_list):
    if isinstance(ingredients_list, str):
        ingredients_list = parse_ingredients(ingredients_list)
    return ' '.join(ingredients_list)


def build_tfidf_matrix(df, column='Normalized_Ingredients'):
    ingredient_texts = df[column].apply(ingredients_to_text)
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words=None
    )
    
    tfidf_matrix = vectorizer.fit_transform(ingredient_texts)
    
    return tfidf_matrix, vectorizer, ingredient_texts


def train_kmeans(tfidf_matrix, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    return kmeans, clusters


def analyze_clusters(df, clusters, n_clusters=20):
    print("\n" + "="*80)
    print("clusters")
    print("="*80)
    
    for i in range(n_clusters):
        cluster_recipes = df[clusters == i]
        print(f"\nCluster {i}: {len(cluster_recipes)} recipes")
        
        all_ingredients = []
        for idx in cluster_recipes.index:
            ings = parse_ingredients(df.loc[idx, 'Normalized_Ingredients'])
            all_ingredients.extend(ings)
        
        top_ingredients = Counter(all_ingredients).most_common(5)
        print(f"  Top ingredients: {', '.join([f'{ing}({cnt})' for ing, cnt in top_ingredients])}")
        
        print(f"  Sample recipes: {', '.join(cluster_recipes['Title'].head(3).tolist())}")


def compute_pantry_coverage(recipe_ingredients, pantry_ingredients):
    """
    coverage with normalization, lemmatization, and fuzzy matching
    """
    if not recipe_ingredients:
        return 0.0

    # normalize ings
    norm_recipe = [normalize_ingredient(ing) for ing in recipe_ingredients]
    norm_pantry = [normalize_ingredient(ing) for ing in pantry_ingredients]

    matched = 0
    for ing in norm_recipe:
        # exact match
        if ing in norm_pantry:
            matched += 1
        # fuzzy match
        elif fuzzy_match(ing, norm_pantry):
            matched += 1
    coverage = matched / len(norm_recipe)
    return coverage

## penalize simple recipes like sauces, dressings, etc.
def recipe_complexity_score(num_ingredients):
    if num_ingredients <= 3:
        return 0.5 # heavy penalty for sauces/dressings
    elif num_ingredients <= 5:
        return 0.8  # light penalty for side dishes
    else:
        return 1.0 # no penalty for main dishes

## k-means to rank recipes based on affinity
def recommend_recipes(pantry_ingredients, df, tfidf_matrix, vectorizer, kmeans, top_k=10, tfidf_weight=0.3, coverage_weight=0.3, cluster_weight=0.2, sustainability_weight=0.2):

    pantry_text = ' '.join(pantry_ingredients)
    pantry_vector = vectorizer.transform([pantry_text])
    cluster_centroids = kmeans.cluster_centers_
    cluster_similarities = cosine_similarity(pantry_vector, cluster_centroids)[0]

    cluster_scores = cluster_similarities[df['cluster'].values]
    tfidf_similarities = cosine_similarity(pantry_vector, tfidf_matrix)[0]
    
    # coverage, complexity and sustainability scores for each recipe
    coverages, complexity_scores, sustainability_emissions = compute_recipe_metrics(df, pantry_ingredients)
    
    coverages = np.array(coverages)
    complexity_scores = np.array(complexity_scores)
    sustainability_emissions = np.array(sustainability_emissions)
    
    # inverse transform, robust to outliers
    sustainability_scores = 1.0 / (1.0 + sustainability_emissions)

    base_scores = (tfidf_weight * tfidf_similarities + 
                   coverage_weight * coverages + 
                   cluster_weight * cluster_scores +
                   sustainability_weight * sustainability_scores)
    
    # apply penalty
    scores = base_scores * complexity_scores
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = tfidf_similarities[top_indices]
    results['pantry_coverage'] = coverages[top_indices]
    results['cluster_score'] = cluster_scores[top_indices]
    results['complexity_score'] = complexity_scores[top_indices]
    results['base_score'] = base_scores[top_indices]
    results['combined_score'] = scores[top_indices]
    results['sustainability_emissions'] = sustainability_emissions[top_indices]
    
    return results[['Title', 'cluster', 'Normalized_Ingredients', 'similarity_score', 
                    'pantry_coverage', 'cluster_score', 'complexity_score', 'combined_score',
                    'sustainability_emissions']]


def visualize_clusters(tfidf_matrix, clusters, n_clusters=20):
    from sklearn.decomposition import PCA
    
    sample_size = min(5000, tfidf_matrix.shape[0])
    sample_idx = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix[sample_idx].toarray())
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=clusters[sample_idx], cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Recipe Clusters (K-Means, k={n_clusters})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.tight_layout()
    plt.savefig('cluster_visualization.png', dpi=150)


def main():
    data_path = Path("../data/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv")
    df = load_data(data_path)
    
    tfidf_matrix, vectorizer, ingredient_texts = build_tfidf_matrix(df)
    
    n_clusters = 20
    kmeans, clusters = train_kmeans(tfidf_matrix, n_clusters=n_clusters)

    df['cluster'] = clusters
    
    analyze_clusters(df, clusters, n_clusters=n_clusters)
    
    visualize_clusters(tfidf_matrix, clusters, n_clusters=n_clusters)
    
    ## pantry for testing
    pantry = ["potato", "carrot", "onion", "garlic",
 "olive oil", "vegetable oil", "butter",
 "salt", "black pepper", "flour", "sugar",
 "soy sauce", "vinegar", "ketchup", "mayonnaise",
 "rice", "sugar", "steak", "pepper"]
    
    print(f"\nPantry ingredients: {pantry}")
    
    # Get recommendations using cluster scoring
    recommendations = recommend_recipes(
        pantry, df, tfidf_matrix, vectorizer, kmeans, top_k=10
    )
    
    print("\n" + "="*80)
    print("top 10 rec (k_means)")
    print("="*80)
    for idx, row in recommendations.iterrows():
        print(f"\n{row['Title']} [Cluster {row['cluster']}]")
        print(f"  TF-IDF Similarity: {row['similarity_score']:.3f}")
        print(f"  Pantry Coverage:   {row['pantry_coverage']:.1%}")
        print(f"  Cluster Score:     {row['cluster_score']:.3f}")
        print(f"  Complexity Score:  {row['complexity_score']:.2f}")
        print(f"  Combined Score:    {row['combined_score']:.3f}")
        if 'sustainability_emissions' in row:
            try:
                print(f"  Sustainability (kg CO2e): {row['sustainability_emissions']:.2f}")
            except Exception:
                print(f"  Sustainability (kg CO2e): {row['sustainability_emissions']}")
        ings = parse_ingredients(row['Normalized_Ingredients'])
        print(f"  Ingredients ({len(ings)}): {', '.join(ings)}")
    
    evaluator = ClusteringEvaluator(tfidf_matrix, clusters, kmeans)

    print("\n=== ELBOW ANALYSIS ===")
    elbow_results = evaluator.elbow_analysis(max_k=30)
    evaluator.plot_analysis(elbow_results, current_k=n_clusters)

    metrics = evaluator.calculate_metrics()
    
    return df, tfidf_matrix, vectorizer, kmeans, recommendations, metrics


if __name__ == "__main__":
    df, tfidf_matrix, vectorizer, kmeans, recommendations, metrics = main()
