"""
Recipe Recommendation Pipeline
4-stage system: Retrieval -> Filtering -> ML Scoring -> Ranking
"""

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
import re
import ssl
import ast

# NLTK setup
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.stem import WordNetLemmatizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

ALIASES = {
    'scallion': 'green onion',
    'spring onion': 'green onion',
    'roma tomato': 'tomato',
    'plum tomato': 'tomato',
    'cherry tomato': 'tomato',
}

def normalize_ingredient(ingredient):
    """Normalize ingredient name"""
    ingredient = ingredient.lower().strip()
    ingredient = re.sub(r'\d+\.?\d*\s*(cup|tablespoon|teaspoon|tbsp|tsp|oz|ounce|pound|lb|gram|g|ml|liter|l)s?', '', ingredient)
    ingredient = re.sub(r'\d+\.?\d*', '', ingredient)
    descriptors = [
        'fresh', 'frozen', 'canned', 'dried', 'chopped', 'diced', 'sliced',
        'minced', 'crushed', 'ground', 'whole', 'optional', 'to taste',
        'boneless', 'skinless', 'bone in', 'boned', 'uncooked', 'cooked',
        'low sodium', 'reduced sodium', 'unsalted', 'no salt', 'reduced fat',
        'low fat', 'fat free', 'light', 'reduced', 'organic'
    ]
    for desc in descriptors:
        ingredient = ingredient.replace(desc, '')
    ingredient = re.sub(r'[^\w\s]', ' ', ingredient)
    ingredient = re.sub(r'\s+', ' ', ingredient).strip()
    words = ingredient.split()
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in words])
    alias_map = {
        **ALIASES,
        'chicken breast': 'chicken',
        'chicken thighs': 'chicken',
        'ground beef': 'beef',
        'lean ground beef': 'beef',
        'ground turkey': 'turkey',
        'ground pork': 'pork',
        'pork chop': 'pork',
        'salmon fillet': 'salmon',
        'white sugar': 'sugar',
        'brown sugar': 'sugar',
        'all purpose flour': 'flour',
    }
    for alias, canonical in alias_map.items():
        if alias in lemmatized:
            lemmatized = lemmatized.replace(alias, canonical)
    return lemmatized

class RecipeRecommender:
    """Multi-task learning recipe recommender"""
    
    def __init__(self, data_path, vectorizer_path, model_dir):
        """Load data, vectorizer, and models"""
        print("Loading recommender system...")
        
        # Load data
        self.df = pd.read_csv(data_path)
        print(f"  Loaded {len(self.df)} recipes")
        
        # Parse ingredients if needed
        if isinstance(self.df['normalized_ingredients'].iloc[0], str):
            self.df['normalized_ingredients'] = self.df['normalized_ingredients'].apply(ast.literal_eval)
        
        # Load TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"  Loaded TF-IDF vectorizer")
        
        # Compute TF-IDF matrix for all recipes
        self.tfidf_matrix = self.vectorizer.transform(self.df['ingredient_text'])
        print(f"  TF-IDF matrix: {self.tfidf_matrix.shape}")
        
        # Load models
        self.model_taste = lgb.Booster(model_file=f'{model_dir}/model_taste.txt')
        self.model_nutrition = lgb.Booster(model_file=f'{model_dir}/model_nutrition.txt')
        self.model_sustainability = lgb.Booster(model_file=f'{model_dir}/model_sustainability.txt')
        self.diet_models = {
            'vegan': lgb.Booster(model_file=f'{model_dir}/diet_vegan.txt'),
            'vegetarian': lgb.Booster(model_file=f'{model_dir}/diet_vegetarian.txt'),
            'gluten_free': lgb.Booster(model_file=f'{model_dir}/diet_gluten_free.txt'),
            'low_sodium': lgb.Booster(model_file=f'{model_dir}/diet_low_sodium.txt'),
            'low_sugar': lgb.Booster(model_file=f'{model_dir}/diet_low_sugar.txt'),
            'healthy': lgb.Booster(model_file=f'{model_dir}/diet_healthy.txt')
        }
        print(f"  Loaded 3 models from {model_dir}/")
        
        print("✓ Recommender ready!\n")
    
    def recommend(self, pantry_items, intent='main', coverage_threshold=0.7, 
                  top_k=10, weights={'taste': 0.4, 'nutrition': 0.3, 'sustainability': 0.3, 'diet': 0.0},
                  allow_sauces=False, diet_preference=None, diet_threshold=0.5):
        """
        4-stage recommendation pipeline
        
        Args:
            pantry_items: List of available ingredients
            intent: 'main' (>=7 ingredients) or 'side' (<7 ingredients)
            coverage_threshold: Minimum pantry coverage (0-1)
            top_k: Number of recommendations
            weights: Multi-objective weights (taste/nutrition/sustainability/diet)
                - Set diet weight > 0 to include diet score in ranking
                - If diet_preference is set, uses that specific diet model
            allow_sauces: If False, sauces/condiments/marinades are removed from main intent
            diet_preference: one of ['vegan','vegetarian','gluten_free','low_sodium','low_sugar','healthy']
                - If set with diet weight > 0: includes diet probability in final score
                - If set with diet_threshold > 0: filters recipes below threshold (hard filter)
            diet_threshold: minimum probability to keep (0 = no filtering, 0.5 = moderate, 0.8 = strict)
        """
        
        print("="*70)
        print(f"RECIPE RECOMMENDATION (intent={intent}, coverage>={coverage_threshold})")
        print("="*70)
        
        # Normalize pantry items
        pantry_normalized = [normalize_ingredient(item) for item in pantry_items]
        pantry_text = ' '.join(pantry_normalized)
        print(f"\n[Pantry] {len(pantry_items)} items: {pantry_items[:5]}...")
        print(f"[Normalized] {pantry_normalized[:5]}...")
        pantry_set = set(pantry_normalized)
        
        def matches_pantry(ingredient):
            """Loose matcher to avoid missing overlaps from wording differences"""
            if ingredient in pantry_set:
                return True
            for p in pantry_set:
                if len(p) >= 3 and (ingredient in p or p in ingredient):
                    return True
            return False
        
        # ========== STAGE 1: RETRIEVAL (TF-IDF + Coverage) ==========
        print("\n[STAGE 1] Retrieval: TF-IDF similarity + Coverage filter")
        
        # Compute TF-IDF similarity
        pantry_vector = self.vectorizer.transform([pantry_text])
        similarities = cosine_similarity(pantry_vector, self.tfidf_matrix).flatten()
        
        # Calculate pantry coverage
        def calc_coverage(recipe_ingredients):
            if len(recipe_ingredients) == 0:
                return 0.0
            matched = sum(1 for ing in recipe_ingredients if matches_pantry(ing))
            return matched / len(recipe_ingredients)
        
        self.df['pantry_coverage'] = self.df['normalized_ingredients'].apply(calc_coverage)
        self.df['tfidf_similarity'] = similarities
        
        # Filter by coverage threshold
        candidates = self.df[self.df['pantry_coverage'] >= coverage_threshold].copy()
        print(f"  Candidates after coverage filter (>={coverage_threshold}): {len(candidates)}")
        
        if len(candidates) == 0:
            print(f"  ❌ No recipes found with coverage >= {coverage_threshold}")
            return pd.DataFrame()
        
        # ========== STAGE 2: INTENT FILTERING ==========
        print(f"\n[STAGE 2] Intent filter: {intent} dish")
        
        if intent == 'main':
            candidates = candidates[candidates['n_ingredients'] >= 7]
            # Trim extremely quick/simple items that are usually sauces/condiments
            candidates = candidates[candidates['minutes'] >= 10]
            print(f"  Filtered to main dishes (>=7 ingredients, >=10 min): {len(candidates)}")
        elif intent == 'side':
            candidates = candidates[candidates['n_ingredients'] < 7]
            print(f"  Filtered to side dishes (<7 ingredients): {len(candidates)}")
        
        # Remove obvious sauces/condiments for mains unless explicitly allowed
        def is_sauce_like(name):
            name = name.lower()
            keywords = [
                'sauce', 'dressing', 'marinade', 'gravy', 'salsa', 'dip', 'spread',
                'marinara', 'pasta sauce', 'pizza sauce', 'spaghetti sauce'
            ]
            return any(k in name for k in keywords)
        
        if intent == 'main' and not allow_sauces:
            before = len(candidates)
            candidates = candidates[~candidates['name'].apply(is_sauce_like)]
            print(f"  Removed sauces/condiments for mains: {before} -> {len(candidates)}")
        
        if len(candidates) == 0:
            print(f"  ❌ No {intent} dishes found")
            return pd.DataFrame()
        
        # ========== STAGE 3: ML SCORING ==========
        print(f"\n[STAGE 3] ML Prediction: 3 objectives")
        
        # Prepare features
        # Taste: n_ingredients, minutes, n_steps, avg_ingredient_commonality
        X_taste = candidates[['n_ingredients', 'minutes', 'n_steps', 'avg_ingredient_commonality']].values
        
        # Nutrition: structured + SVD ingredient signals (no diet flags to avoid leakage)
        svd_cols = [c for c in candidates.columns if c.startswith('tfidf_svd_')]
        nutr_cols = [
            'protein_density', 'sugar_density', 'sodium_density',
            'fat', 'sat_fat', 'carbs', 'calories', 'n_ingredients',
            'avg_ingredient_commonality', 'plant_ratio', 'meat_ratio'
        ] + svd_cols
        X_nutrition = candidates[nutr_cols].fillna(0).values
        
        # Sustainability: protein, fat, n_ingredients, avg_ingredient_commonality
        X_sustainability = candidates[['protein', 'fat', 'n_ingredients', 'avg_ingredient_commonality']].values
        
        # Diet: CRITICAL - Exclude plant_ratio/meat_ratio (label leakage)
        # Use TF-IDF + nutrition only, forcing model to learn ingredient semantics
        diet_cols = [
            'protein_density', 'sugar_density', 'sodium_density',
            'fat', 'sat_fat', 'carbs', 'calories', 'n_ingredients',
            'avg_ingredient_commonality'
        ] + svd_cols
        X_diet = candidates[diet_cols].fillna(0).values
        
        # Predict scores
        candidates['taste_score'] = self.model_taste.predict(X_taste)
        candidates['nutrition_score_pred'] = self.model_nutrition.predict(X_nutrition)
        candidates['sustainability_score_pred'] = self.model_sustainability.predict(X_sustainability)
        # Diet probabilities
        diet_probs = {}
        for name, model in self.diet_models.items():
            diet_probs[name] = model.predict(X_diet)
            candidates[f'diet_{name}'] = diet_probs[name]
        
        print(f"  Predicted scores:")
        print(f"    Taste: {candidates['taste_score'].mean():.3f} ± {candidates['taste_score'].std():.3f}")
        print(f"    Nutrition: {candidates['nutrition_score_pred'].mean():.3f} ± {candidates['nutrition_score_pred'].std():.3f}")
        print(f"    Sustainability: {candidates['sustainability_score_pred'].mean():.3f} ± {candidates['sustainability_score_pred'].std():.3f}")
        
        # ========== STAGE 4: MULTI-OBJECTIVE RANKING ==========
        print(f"\n[STAGE 4] Multi-objective ranking")
        print(f"  Weights: Taste={weights['taste']}, Nutrition={weights['nutrition']}, Sustainability={weights['sustainability']}, Diet={weights['diet']}")
        if diet_preference:
            print(f"  Diet preference: {diet_preference} (threshold >= {diet_threshold})")
        
        # Normalize scores to [0, 1]
        # Taste: avg_rating is 1-5, normalize to 0-1
        candidates['taste_norm'] = (candidates['taste_score'] - 1) / 4.0
        candidates['nutrition_norm'] = candidates['nutrition_score_pred']  # Already 0-1
        candidates['sustainability_norm'] = candidates['sustainability_score_pred']  # Already 0-1
        
        # Diet: use specified diet preference or default to 0
        diet_weight = weights.get('diet', 0.0)
        if diet_weight > 0 and diet_preference:
            pref_col = f'diet_{diet_preference}'
            if pref_col in candidates.columns:
                candidates['diet_norm'] = candidates[pref_col]  # Already 0-1 (probability)
            else:
                print(f"  ⚠️  Warning: diet column '{pref_col}' not found, setting diet_norm=0")
                candidates['diet_norm'] = 0.0
        else:
            candidates['diet_norm'] = 0.0
        
        # Pantry coverage / TF-IDF often have tight ranges per query; scale per-query
        def minmax(series):
            smin, smax = series.min(), series.max()
            if smax == smin:
                return pd.Series(0.5, index=series.index)
            return (series - smin) / (smax - smin)
        candidates['coverage_norm'] = minmax(candidates['pantry_coverage'])
        candidates['tfidf_norm'] = minmax(candidates['tfidf_similarity'])
        
        # Combine scores (4 ML models + 2 retrieval bonuses)
        coverage_bonus = 0.20
        tfidf_bonus = 0.10
        total_ml_weight = weights['taste'] + weights['nutrition'] + weights['sustainability'] + diet_weight
        total_weight = total_ml_weight + coverage_bonus + tfidf_bonus
        
        candidates['final_score'] = (
            weights['taste'] * candidates['taste_norm'] +
            weights['nutrition'] * candidates['nutrition_norm'] +
            weights['sustainability'] * candidates['sustainability_norm'] +
            diet_weight * candidates['diet_norm'] +
            coverage_bonus * candidates['coverage_norm'] +
            tfidf_bonus * candidates['tfidf_norm']
        ) / total_weight
        
        # Apply diet hard filter if threshold > 0
        if diet_preference and diet_threshold > 0:
            pref_col = f'diet_{diet_preference}'
            if pref_col in candidates.columns:
                before_diet = len(candidates)
                candidates = candidates[candidates[pref_col] >= diet_threshold]
                print(f"  Diet hard filter ({diet_preference} >= {diet_threshold}): {before_diet} -> {len(candidates)}")
                if len(candidates) == 0:
                    print(f"  ⚠️  No recipes meet diet threshold! Consider lowering diet_threshold or diet weight.")
        
        # Sort by final score
        recommendations = candidates.sort_values('final_score', ascending=False).head(top_k)
        
        print(f"\n✓ Top {min(top_k, len(recommendations))} Recommendations:")
        print("="*70)
        
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{idx}. {row['name']}")
            print(f"   Coverage: {row['pantry_coverage']:.1%} | Ingredients: {row['n_ingredients']} | Time: {row['minutes']}min")
            print(f"   Taste: {row['taste_score']:.2f}/5.0 | Nutrition: {row['nutrition_score_pred']:.3f} | Sustainability: {row['sustainability_score_pred']:.3f}")
            print(f"   Final Score: {row['final_score']:.3f}")
        
        return recommendations[[
            'name', 'n_ingredients', 'minutes', 'pantry_coverage',
            'tfidf_similarity', 'taste_score', 'nutrition_score_pred',
            'sustainability_score_pred', 'final_score'
        ]]

def load_recommender(data_path='../data/recipes_with_features.csv',
                     vectorizer_path='../data/tfidf_vectorizer.pkl',
                     model_dir='./models'):
    """Convenience function to load recommender"""
    return RecipeRecommender(data_path, vectorizer_path, model_dir)

if __name__ == '__main__':
    # Example usage
    print("\n" + "="*70)
    print("EXAMPLE: Recipe Recommendation Demo")
    print("="*70)
    
    # Load recommender
    recommender = load_recommender()
    
    # Example pantry
    pantry = [
        'chicken breast', 'olive oil', 'garlic', 'onion', 'tomato',
        'pasta', 'parmesan cheese', 'basil', 'salt', 'pepper',
        'rice', 'soy sauce', 'ginger', 'carrot', 'broccoli'
    ]
    
    # Get recommendations
    print("\n" + "="*70)
    print("MAIN DISH RECOMMENDATIONS")
    recommendations_main = recommender.recommend(
        pantry_items=pantry,
        intent='main',
        coverage_threshold=0.7,
        top_k=5
    )
    
    print("\n" + "="*70)
    print("SIDE DISH RECOMMENDATIONS")
    recommendations_side = recommender.recommend(
        pantry_items=pantry,
        intent='side',
        coverage_threshold=0.7,
        top_k=5
    )
