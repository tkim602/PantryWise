import ast
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def alias_map(token):
    return ALIASES.get(token, token)

def sustainability_score(ings):
    # Simple additive score; normalize later across dataset
    # A better way to do this would probably be checking if there is data about carbon footprint per kg of ingredient (?)
    # then do a proper lookup instead of pre-defined dictionary to compute an ingredient quantity weighted score.
    return float(sum(SUSTAINABILITY.get(x, 0.0) for x in ings))

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


def compute_pantry_coverage(recipe_ingredients, pantry_ingredients):
    if not recipe_ingredients:
        return 0.0
    
    recipe_set = set(recipe_ingredients)
    pantry_set = set(pantry_ingredients)
    
    overlap = len(recipe_set & pantry_set)
    coverage = overlap / len(recipe_set)
    
    return coverage


def recommend_recipes(pantry_ingredients, df, tfidf_matrix, vectorizer, 
                     top_k=10, coverage_weight=0.3, similarity_weight=0.7):
    pantry_text = ' '.join(pantry_ingredients)
    pantry_vector = vectorizer.transform([pantry_text])
    
    similarities = cosine_similarity(pantry_vector, tfidf_matrix)[0]
    
    coverages = []
    for idx, row in df.iterrows():
        recipe_ings = parse_ingredients(row['Normalized_Ingredients'])
        coverage = compute_pantry_coverage(recipe_ings, pantry_ingredients)
        coverages.append(coverage)
    
    coverages = np.array(coverages)
    scores = (similarity_weight * similarities) + (coverage_weight * coverages)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    results['pantry_coverage'] = coverages[top_indices]
    results['combined_score'] = scores[top_indices]
    
    return results[['Title', 'Normalized_Ingredients', 'similarity_score', 
                    'pantry_coverage', 'combined_score']]


def main():
    data_path = Path("../data/food-ingredients-and-recipe-dataset-with-images/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv")
    df = load_data(data_path)
    
    tfidf_matrix, vectorizer, ingredient_texts = build_tfidf_matrix(df)
    
    pantry = ["potato", "carrot", "onion", "garlic",
 "olive oil", "vegetable oil", "butter",
 "salt", "black pepper", "flour", "sugar",
 "soy sauce", "vinegar", "ketchup", "mayonnaise"]
    
    print(f"\npantry: {pantry}")
    
    recommendations = recommend_recipes(pantry, df, tfidf_matrix, vectorizer, top_k=10)
    
    print("\n" + "="*80)
    print("top 10 rec (baseline)")
    print("="*80)
    for idx, row in recommendations.iterrows():
        print(f"\n{row['Title']}")
        print(f"  Similarity: {row['similarity_score']:.3f}")
        print(f"  Coverage:   {row['pantry_coverage']:.1%}")
        print(f"  Score:      {row['combined_score']:.3f}")
        ings = parse_ingredients(row['Normalized_Ingredients'])
        print(f"  Ingredients: {', '.join(ings)}")
    
    return df, tfidf_matrix, vectorizer, recommendations


if __name__ == "__main__":
    df, tfidf_matrix, vectorizer, recommendations = main()
