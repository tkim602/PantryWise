"""
Data Preprocessing for Food.com Dataset
Loads raw data, processes fields, calculates objective scores
"""

import pandas as pd
import numpy as np
import ast
import re
import ssl
import pickle
from pathlib import Path

# NLTK setup
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
except ImportError:
    print("Installing NLTK...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

# SSL workaround for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Ingredient aliases
ALIASES = {
    'scallion': 'green onion',
    'spring onion': 'green onion',
    'roma tomato': 'tomato',
    'plum tomato': 'tomato',
    'cherry tomato': 'tomato',
    'grape tomato': 'tomato',
}

# Diet keyword helpers
ANIMAL_KEYWORDS = ['beef', 'steak', 'veal', 'lamb', 'pork', 'bacon', 'ham', 'sausage',
                   'chicken', 'turkey', 'poultry', 'fish', 'salmon', 'tuna', 'shrimp',
                   'egg', 'cheese', 'milk', 'butter', 'cream', 'yogurt']
DAIRY_KEYWORDS = ['milk', 'cheese', 'butter', 'cream', 'yogurt']
GLUTEN_KEYWORDS = ['wheat', 'barley', 'rye', 'flour', 'bread', 'pasta']
TAG_HEALTHY = {'healthy', 'healthy-2'}

def normalize_ingredient(ingredient):
    """Normalize ingredient name (from model.py)"""
    # Lowercase
    ingredient = ingredient.lower().strip()
    
    # Remove quantities and units up front
    ingredient = re.sub(r'\d+\.?\d*\s*(cup|tablespoon|teaspoon|tbsp|tsp|oz|ounce|pound|lb|gram|g|ml|liter|l)s?', '', ingredient)
    ingredient = re.sub(r'\(\s*\d+\.?\d*\s*oz\s*\)', '', ingredient)
    ingredient = re.sub(r'\d+\.?\d*', '', ingredient)
    
    # Normalize descriptors that otherwise break pantry matching
    descriptors = [
        'fresh', 'frozen', 'canned', 'dried', 'chopped', 'diced', 'sliced',
        'minced', 'crushed', 'ground', 'whole', 'optional', 'to taste',
        'boneless', 'skinless', 'bone in', 'boned', 'uncooked', 'cooked',
        'low sodium', 'reduced sodium', 'unsalted', 'no salt', 'reduced fat',
        'low fat', 'fat free', 'light', 'reduced', 'organic'
    ]
    for desc in descriptors:ㅇㅇ
        ingredient = ingredient.replace(desc, '')
    
    # Clean up
    ingredient = re.sub(r'[^\w\s]', ' ', ingredient)
    ingredient = re.sub(r'\s+', ' ', ingredient).strip()
    
    # Lemmatize
    words = ingredient.split()
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in words])
    
    # Apply aliases
    # Also collapse common meat/dairy phrases to improve coverage matching
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

def calculate_nutrition_score(row):
    """Calculate nutrition quality score (0-1)"""
    try:
        protein = float(row['protein']) if pd.notna(row['protein']) else 0
        fiber = float(row.get('fiber', 0)) if 'fiber' in row and pd.notna(row.get('fiber')) else 0
        sodium = float(row['sodium']) if pd.notna(row['sodium']) else 0
        sugar = float(row['sugar']) if pd.notna(row['sugar']) else 0
        
        # Higher protein/fiber = good, lower sodium/sugar = good
        score = (
            min(protein / 30.0, 1.0) * 0.3 +      # Up to 30g protein
            min(fiber / 10.0, 1.0) * 0.3 +        # Up to 10g fiber
            max(0, 1 - sodium / 2000.0) * 0.2 +   # Sodium limit 2000mg
            max(0, 1 - sugar / 50.0) * 0.2        # Sugar limit 50g
        )
        return max(0, min(1, score))
    except:
        return 0.5  # Default neutral

def calculate_sustainability_with_carbon(ingredients):
    """
    Calculate sustainability score AND estimated carbon footprint
    Returns: (score, avg_carbon)
    """
    score = calculate_sustainability_score(ingredients)
    
    # Calculate avg carbon for features
    if not ingredients or len(ingredients) == 0:
        return (score, 2.0)  # Default medium carbon
    
    CARBON_FOOTPRINT = {
        'beef': 60.0, 'steak': 60.0, 'veal': 45.0, 'lamb': 40.0, 'mutton': 40.0,
        'pork': 7.0, 'bacon': 7.5, 'ham': 7.0, 'sausage': 8.0, 'cheese': 9.0,
        'butter': 5.0, 'cream': 4.5, 'prawn': 5.0, 'shrimp': 5.0,
        'chicken': 2.5, 'turkey': 2.5, 'poultry': 2.5,
        'salmon': 3.0, 'tuna': 3.5, 'fish': 2.5,
        'milk': 1.0, 'yogurt': 1.0, 'egg': 1.5,
        'tofu': 0.5, 'lentil': 0.4, 'bean': 0.4, 'pea': 0.4,
        'potato': 0.2, 'rice': 0.5, 'pasta': 0.3, 'bread': 0.4,
        'tomato': 0.3, 'carrot': 0.2, 'onion': 0.2, 'garlic': 0.2,
        'broccoli': 0.2, 'spinach': 0.2, 'mushroom': 0.3,
        'flour': 0.3, 'sugar': 0.5, 'oil': 2.0
    }
    
    total_carbon = 0
    for ing in ingredients:
        ing_lower = ing.lower()
        carbon = None
        for key, value in CARBON_FOOTPRINT.items():
            if key in ing_lower:
                carbon = value
                break
        if carbon is None:
            carbon = 0.5  # Default plant-based
        total_carbon += carbon
    
    avg_carbon = total_carbon / len(ingredients)
    return (score, avg_carbon)

def calculate_sustainability_score(ingredients):
    """
    Calculate sustainability score based on REAL carbon emission data (0-1)
    Based on research: kg CO2 equivalent per kg of food product
    
    Sources:
    - Poore & Nemecek (2018) Science
    - Our World in Data carbon footprint database
    """
    if not ingredients or len(ingredients) == 0:
        return 0.5
    
    # Carbon footprint database (kg CO2eq per kg food)
    CARBON_FOOTPRINT = {
        # Very High Carbon (>10 kg CO2/kg)
        'beef': 60.0, 'steak': 60.0, 'veal': 45.0, 
        'lamb': 40.0, 'mutton': 40.0,
        
        # High Carbon (5-10 kg CO2/kg)
        'pork': 7.0, 'bacon': 7.5, 'ham': 7.0, 
        'sausage': 8.0, 'cheese': 9.0,
        
        # Medium-High Carbon (3-5 kg CO2/kg)
        'butter': 5.0, 'cream': 4.5, 'prawn': 5.0, 'shrimp': 5.0,
        
        # Medium Carbon (1-3 kg CO2/kg)
        'chicken': 2.5, 'turkey': 2.5, 'poultry': 2.5,
        'salmon': 3.0, 'tuna': 3.5, 'fish': 2.5,
        
        # Low Carbon (0.5-1 kg CO2/kg)
        'milk': 1.0, 'yogurt': 1.0, 'egg': 1.5,
        
        # Very Low Carbon (<0.5 kg CO2/kg) - Plant-based
        'tofu': 0.5, 'lentil': 0.4, 'bean': 0.4, 'pea': 0.4,
        'potato': 0.2, 'rice': 0.5, 'pasta': 0.3, 'bread': 0.4,
        'tomato': 0.3, 'carrot': 0.2, 'onion': 0.2, 'garlic': 0.2,
        'broccoli': 0.2, 'spinach': 0.2, 'mushroom': 0.3,
        'apple': 0.2, 'banana': 0.3, 'orange': 0.3,
        'flour': 0.3, 'sugar': 0.5, 'oil': 2.0, 'olive oil': 2.5,
        'nut': 0.3, 'seed': 0.3, 'grain': 0.3, 'quinoa': 0.5
    }
    
    # Default values for unmatched ingredients
    DEFAULT_ANIMAL = 4.0  # Assume medium animal product
    DEFAULT_PLANT = 0.3   # Assume plant-based
    
    total_carbon = 0
    matched_count = 0
    
    for ing in ingredients:
        ing_lower = ing.lower()
        
        # Find matching carbon value
        carbon = None
        for key, value in CARBON_FOOTPRINT.items():
            if key in ing_lower:
                carbon = value
                break
        
        # If no match, use heuristic
        if carbon is None:
            # Check if likely animal product
            animal_keywords = ['meat', 'dairy', 'milk', 'cheese', 'cream', 'butter', 
                              'egg', 'fish', 'chicken', 'beef', 'pork', 'turkey']
            if any(kw in ing_lower for kw in animal_keywords):
                carbon = DEFAULT_ANIMAL
            else:
                carbon = DEFAULT_PLANT
        
        total_carbon += carbon
        matched_count += 1
    
    if matched_count == 0:
        return 0.5
    
    # Average carbon footprint per ingredient
    avg_carbon = total_carbon / matched_count
    
    # Convert to 0-1 score (lower carbon = higher score)
    # Reference scale:
    #   0 kg CO2/kg = 1.0 (perfect)
    #   10 kg CO2/kg = 0.5 (medium)
    #   60 kg CO2/kg = 0.0 (worst - pure beef)
    
    # Using logarithmic scale for better distribution
    # log(1 + x) to handle 0 values
    import math
    log_carbon = math.log(1 + avg_carbon)
    log_max = math.log(1 + 60)  # Max: pure beef
    
    score = 1.0 - (log_carbon / log_max)
    
    return max(0, min(1, score))

def compute_plant_meat_ratio(ingredients):
    """Approximate plant vs animal ingredient ratio"""
    if not ingredients:
        return (0.0, 0.0)
    animal = 0
    for ing in ingredients:
        ing_lower = ing.lower()
        if any(kw in ing_lower for kw in ANIMAL_KEYWORDS):
            animal += 1
    total = len(ingredients)
    meat_ratio = animal / total
    plant_ratio = 1.0 - meat_ratio
    return (plant_ratio, meat_ratio)

def compute_diet_flags(tags_list, ingredients, row):
    """
    Return dict of diet flags: vegan, vegetarian, gluten_free, low_sodium, low_sugar
    
    IMPROVED LOGIC FOR ML:
    - Use BOTH tags AND ingredients (tags are noisy, ingredients are ground truth)
    - Strict animal keyword matching
    - Conservative labeling (if uncertain, set to -1 for exclusion during training)
    """
    tags_lower = [t.lower() for t in tags_list] if isinstance(tags_list, list) else []
    ing_lower = [ing.lower() for ing in ingredients] if isinstance(ingredients, list) else []
    
    # Meat keywords (strict matching)
    MEAT_KEYWORDS = ['beef', 'steak', 'veal', 'lamb', 'pork', 'bacon', 'ham', 'sausage',
                     'chicken', 'turkey', 'poultry', 'duck', 'goose', 
                     'fish', 'salmon', 'tuna', 'shrimp', 'crab', 'lobster', 'seafood']
    
    # Check ingredients for animal products
    has_meat = any(any(kw in ing for kw in MEAT_KEYWORDS) for ing in ing_lower)
    has_dairy = any(any(kw in ing for kw in ['milk', 'cheese', 'butter', 'cream', 'yogurt']) for ing in ing_lower)
    has_egg = any('egg' in ing for ing in ing_lower)
    has_gluten = any(any(kw in ing for kw in GLUTEN_KEYWORDS) for ing in ing_lower)
    
    # VEGAN: No animal products at all
    # Rule: Ingredient-based is PRIMARY, tag is SECONDARY confirmation
    vegan_by_ingredients = not (has_meat or has_dairy or has_egg)
    vegan_by_tag = 'vegan' in tags_lower
    
    if vegan_by_ingredients and vegan_by_tag:
        vegan = 1  # Confident vegan
    elif has_meat or has_dairy or has_egg:
        vegan = 0  # Definitely not vegan
    elif vegan_by_tag and not vegan_by_ingredients:
        # Tag says vegan but ingredients suggest otherwise - CONTRADICTION
        vegan = -1  # Exclude from training (noisy label)
    else:
        vegan = 1 if vegan_by_ingredients else 0
    
    # VEGETARIAN: No meat, but allow dairy/eggs
    vegetarian_by_ingredients = not has_meat
    vegetarian_by_tag = 'vegetarian' in tags_lower or vegan_by_tag
    
    if vegetarian_by_ingredients and vegetarian_by_tag:
        vegetarian = 1  # Confident vegetarian
    elif has_meat:
        vegetarian = 0  # Definitely not vegetarian
    elif vegetarian_by_tag and not vegetarian_by_ingredients:
        vegetarian = -1  # Noisy label
    else:
        vegetarian = 1 if vegetarian_by_ingredients else 0
    
    # GLUTEN-FREE
    gluten_free_by_ingredients = not has_gluten
    gluten_free_by_tag = any(t in ['gluten-free', 'gluten free'] for t in tags_lower)
    
    if gluten_free_by_ingredients and gluten_free_by_tag:
        gluten_free = 1
    elif has_gluten:
        gluten_free = 0
    elif gluten_free_by_tag and not gluten_free_by_ingredients:
        gluten_free = -1  # Noisy
    else:
        gluten_free = 1 if gluten_free_by_ingredients else 0
    
    # HEALTHY: Use tag as weak label (subjective)
    healthy_tag = any(t in TAG_HEALTHY for t in tags_lower)
    
    # Low sodium/sugar: Nutrition-based (objective)
    low_sodium = 0
    low_sugar = 0
    try:
        sodium_val = float(row.get('sodium', 0))
        sugar_val = float(row.get('sugar', 0))
        low_sodium = 1 if sodium_val <= 500 else 0
        low_sugar = 1 if sugar_val <= 10 else 0
    except Exception:
        pass
    
    return {
        'flag_vegan': int(vegan),
        'flag_vegetarian': int(vegetarian),
        'flag_gluten_free': int(gluten_free),
        'flag_low_sodium': int(low_sodium),
        'flag_low_sugar': int(low_sugar),
        'flag_healthy': int(healthy_tag)
    }

def preprocess_foodcom_data(data_dir='../data/foodcom_dataset', output_dir='../data'):
    """Main preprocessing pipeline"""
    
    print("="*70)
    print("FOOD.COM DATA PREPROCESSING")
    print("="*70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n[1/6] Loading raw data...")
    recipes = pd.read_csv(f'{data_dir}/RAW_recipes.csv')
    interactions = pd.read_csv(f'{data_dir}/RAW_interactions.csv')
    print(f"  Loaded {len(recipes)} recipes, {len(interactions)} interactions")
    
    # Step 2: Aggregate ratings
    print("\n[2/6] Aggregating ratings (filter: >= 10 reviews)...")
    ratings = interactions.groupby('recipe_id').agg({
        'rating': ['mean', 'std', 'count']
    }).reset_index()
    ratings.columns = ['recipe_id', 'avg_rating', 'rating_std', 'num_reviews']
    ratings = ratings[ratings['num_reviews'] >= 10]
    print(f"  {len(ratings)} recipes with >= 10 reviews")
    
    # Merge
    recipes = recipes.merge(ratings, left_on='id', right_on='recipe_id', how='inner')
    print(f"  After merge: {len(recipes)} recipes")
    
    # Filter outlier cooking times (5 min to 8 hours)
    print(f"  Before time filter: {len(recipes)} recipes")
    print(f"  Time range: {recipes['minutes'].min():.0f} - {recipes['minutes'].max():.0f} minutes")
    recipes = recipes[(recipes['minutes'] >= 5) & (recipes['minutes'] <= 480)]
    print(f"  After time filter (5-480 min): {len(recipes)} recipes")
    
    # Step 3: Parse structured fields
    print("\n[3/6] Parsing structured fields...")
    
    # Parse ingredients (handle NaN)
    print("  Parsing ingredients...")
    recipes['ingredients_list'] = recipes['ingredients'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    
    # Parse nutrition: [calories, fat, sugar, sodium, protein, sat_fat, carbs]
    print("  Parsing nutrition...")
    recipes['nutrition_list'] = recipes['nutrition'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else [0]*7
    )
    nutrition_df = pd.DataFrame(recipes['nutrition_list'].tolist(), 
                                 columns=['calories', 'fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs'])
    recipes = pd.concat([recipes, nutrition_df], axis=1)
    
    # Parse tags (handle NaN)
    print("  Parsing tags...")
    recipes['tags_list'] = recipes['tags'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    
    # Step 4: Normalize ingredients
    print("\n[4/6] Normalizing ingredients...")
    recipes['normalized_ingredients'] = recipes['ingredients_list'].apply(
        lambda lst: [normalize_ingredient(ing) for ing in lst if isinstance(ing, str) and ing.strip()] if isinstance(lst, list) else []
    )
    recipes['ingredient_text'] = recipes['normalized_ingredients'].apply(lambda x: ', '.join(x))
    
    # Filter recipes with no ingredients
    recipes = recipes[recipes['normalized_ingredients'].apply(len) > 0]
    print(f"  After filtering empty ingredients: {len(recipes)} recipes")
    print(f"  Example: {recipes.iloc[0]['ingredients_list'][:2]}")
    print(f"  Normalized: {recipes.iloc[0]['normalized_ingredients'][:2]}")
    
    # Step 5: Calculate objective scores
    print("\n[5/6] Calculating nutrition and sustainability scores...")
    recipes['nutrition_score'] = recipes.apply(calculate_nutrition_score, axis=1)
    
    # Calculate sustainability (returns both score and estimated carbon)
    sustainability_results = recipes['normalized_ingredients'].apply(
        lambda ings: calculate_sustainability_with_carbon(ings)
    )
    recipes['sustainability_score'] = sustainability_results.apply(lambda x: x[0])
    recipes['estimated_carbon'] = sustainability_results.apply(lambda x: x[1])
    ratios = recipes['normalized_ingredients'].apply(compute_plant_meat_ratio)
    recipes['plant_ratio'] = ratios.apply(lambda x: x[0])
    recipes['meat_ratio'] = ratios.apply(lambda x: x[1])
    
    # Diet flags
    diet_flags = recipes.apply(
        lambda row: compute_diet_flags(row['tags_list'], row['normalized_ingredients'], row),
        axis=1
    )
    diet_df = pd.DataFrame(list(diet_flags))
    
    # CRITICAL FIX: Align diet_df index with recipes index
    # pd.concat(axis=1) matches by index, not position!
    diet_df.index = recipes.index
    
    recipes = pd.concat([recipes, diet_df], axis=1)
    
    print(f"  Nutrition score - mean: {recipes['nutrition_score'].mean():.3f}, "
          f"std: {recipes['nutrition_score'].std():.3f}")
    print(f"  Sustainability score - mean: {recipes['sustainability_score'].mean():.3f}, "
          f"std: {recipes['sustainability_score'].std():.3f}")
    
    # Step 6: Select and save
    print("\n[6/6] Saving processed data...")
    
    # Select columns
    processed = recipes[[
        'id', 'name', 'minutes', 'n_ingredients', 'n_steps',
        'ingredients_list', 'normalized_ingredients', 'ingredient_text',
        'calories', 'protein', 'fat', 'sugar', 'sodium', 'sat_fat', 'carbs',
        'avg_rating', 'rating_std', 'num_reviews',
        'nutrition_score', 'sustainability_score', 'estimated_carbon',
        'plant_ratio', 'meat_ratio',
        'flag_vegan', 'flag_vegetarian', 'flag_gluten_free', 'flag_low_sodium', 'flag_low_sugar', 'flag_healthy',
        'tags_list'
    ]].copy()
    
    # Save
    output_path = f'{output_dir}/recipes_processed.csv'
    processed.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    print(f"  Total recipes: {len(processed)}")
    
    # Statistics
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Total recipes: {len(processed)}")
    print(f"Average rating: {processed['avg_rating'].mean():.2f} ± {processed['avg_rating'].std():.2f}")
    print(f"Average ingredients: {processed['n_ingredients'].mean():.1f}")
    print(f"Average cooking time: {processed['minutes'].mean():.1f} minutes (range: {processed['minutes'].min():.0f}-{processed['minutes'].max():.0f})")
    print(f"\nCooking time distribution:")
    print(f"  < 30 min: {len(processed[processed['minutes'] < 30])}")
    print(f"  30-60 min: {len(processed[(processed['minutes'] >= 30) & (processed['minutes'] < 60)])}")
    print(f"  60-120 min: {len(processed[(processed['minutes'] >= 60) & (processed['minutes'] < 120)])}")
    print(f"  > 120 min: {len(processed[processed['minutes'] >= 120])}")
    print(f"\nRating distribution:")
    print(f"  5.0 stars: {len(processed[processed['avg_rating'] == 5.0])}")
    print(f"  4.5-5.0: {len(processed[processed['avg_rating'] >= 4.5])}")
    print(f"  4.0-4.5: {len(processed[(processed['avg_rating'] >= 4.0) & (processed['avg_rating'] < 4.5)])}")
    print(f"  < 4.0: {len(processed[processed['avg_rating'] < 4.0])}")
    
    return processed

if __name__ == '__main__':
    processed_data = preprocess_foodcom_data()
    print("\nPreprocessing complete!")
