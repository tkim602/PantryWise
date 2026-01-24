"""
Preprocess Food.com RAW_recipes.csv to create recipes_processed.csv with comma-separated ingredients.

This script reads the raw Food.com dataset and creates a clean CSV with:
- Properly parsed ingredient lists (comma-separated)
- Nutrition per 100g calculations
- Average ratings

Run:
    python model2_lightGBM_v2/preprocess_foodcom.py
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]  # Go up 2 levels: models/ -> model2_lightGBM_v2/ -> project root
DATA_DIR = ROOT / "data"
RAW_RECIPES = DATA_DIR / "foodcom_dataset" / "RAW_recipes.csv"
OUTPUT_PATH = DATA_DIR / "recipes_processed.csv"

# Interactions file for ratings
INTERACTIONS_PATH = DATA_DIR / "foodcom_dataset" / "RAW_interactions.csv"


def parse_list_column(value):
    """Parse string representation of list into actual list."""
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def normalize_ingredient(ingredient: str) -> str:
    """
    Basic ingredient normalization:
    - Lowercase
    - Remove quantities/measurements
    - Remove common descriptors
    - Clean whitespace
    """
    ingredient = ingredient.lower().strip()
    
    # Remove quantities and units
    ingredient = re.sub(r'\d+\.?\d*\s*(cup|tablespoon|teaspoon|tbsp|tsp|oz|ounce|pound|lb|gram|g|ml|liter|l)s?', '', ingredient)
    ingredient = re.sub(r'\(\s*\d+\.?\d*\s*oz\s*\)', '', ingredient)
    ingredient = re.sub(r'\d+\.?\d*', '', ingredient)
    
    # Remove common descriptors
    descriptors = [
        'fresh', 'frozen', 'canned', 'dried', 'chopped', 'diced', 'sliced',
        'minced', 'crushed', 'ground', 'whole', 'optional', 'to taste',
        'boneless', 'skinless', 'low sodium', 'reduced sodium', 'unsalted'
    ]
    for desc in descriptors:
        ingredient = ingredient.replace(desc, '')
    
    # Clean up punctuation and extra spaces
    ingredient = re.sub(r'[^\w\s]', ' ', ingredient)
    ingredient = re.sub(r'\s+', ' ', ingredient).strip()
    
    return ingredient


def calculate_avg_ratings(interactions_path: Path) -> pd.Series:
    """Calculate average rating per recipe from interactions."""
    print("\nCalculating average ratings from interactions...")
    interactions = pd.read_csv(interactions_path)
    avg_ratings = interactions.groupby('recipe_id')['rating'].mean()
    print(f"  Found ratings for {len(avg_ratings):,} recipes")
    return avg_ratings


def preprocess_recipes(raw_path: Path, interactions_path: Path, output_path: Path):
    """Main preprocessing pipeline."""
    print("="*80)
    print("FOOD.COM RECIPE PREPROCESSING")
    print("="*80)
    
    # Load raw recipes
    print(f"\n[1/5] Loading raw recipes from {raw_path.name}...")
    df = pd.read_csv(raw_path)
    print(f"  Loaded {len(df):,} recipes")
    
    # Parse list columns
    print("\n[2/5] Parsing ingredient lists and nutrition...")
    df['ingredients_list'] = df['ingredients'].apply(parse_list_column)
    df['nutrition_list'] = df['nutrition'].apply(parse_list_column)
    
    # Filter recipes with valid ingredients
    df = df[df['ingredients_list'].apply(len) > 0].copy()
    print(f"  After filtering empty ingredients: {len(df):,} recipes")
    
    # Normalize ingredients and create comma-separated text
    print("\n[3/5] Normalizing ingredients...")
    df['normalized_ingredients'] = df['ingredients_list'].apply(
        lambda lst: [normalize_ingredient(ing) for ing in lst if isinstance(ing, str) and ing.strip()]
    )
    
    # ✅ FIX: Use comma + space to join ingredients
    df['ingredient_text'] = df['normalized_ingredients'].apply(lambda x: ', '.join(x))
    
    print(f"  Example original: {df.iloc[0]['ingredients_list'][:3]}")
    print(f"  Example normalized: {df.iloc[0]['normalized_ingredients'][:3]}")
    print(f"  Example text: {df.iloc[0]['ingredient_text'][:100]}...")
    
    # Parse nutrition values (per serving)
    print("\n[4/5] Extracting nutrition per 100g...")
    nutrition_df = pd.DataFrame(df['nutrition_list'].tolist(), 
                                columns=['calories', 'fat', 'sugar', 'sodium', 
                                        'protein', 'sat_fat', 'carbs'])
    
    # Simple per-100g normalization: assume 1 serving = ~150g average
    # This is a rough approximation; ideally we'd use actual serving sizes
    SERVING_SIZE_G = 150
    for col in ['calories', 'fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']:
        nutrition_df[col] = (nutrition_df[col] / SERVING_SIZE_G * 100).round(2)
    
    df = pd.concat([df, nutrition_df], axis=1)
    
    # Calculate average ratings
    avg_ratings = calculate_avg_ratings(interactions_path)
    df['avg_rating'] = df['id'].map(avg_ratings)
    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].median())
    
    # Select output columns
    print("\n[5/5] Saving processed recipes...")
    output_cols = [
        'id', 'name', 'minutes', 'n_steps', 'n_ingredients',
        'ingredient_text', 'calories', 'fat', 'sat_fat', 'carbs',
        'sugar', 'protein', 'sodium', 'avg_rating'
    ]
    
    df[output_cols].to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(df):,} recipes → {output_path}")
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE RECIPE")
    print("="*80)
    sample = df.iloc[0]
    print(f"Name: {sample['name']}")
    print(f"Ingredients ({sample['n_ingredients']}): {sample['ingredient_text'][:150]}...")
    print(f"Nutrition (per 100g): {sample['calories']:.0f} cal, {sample['protein']:.1f}g protein")
    print(f"Rating: {sample['avg_rating']:.2f}/5.0")
    
    print("\n" + "="*80)
    print("✅ Preprocessing complete!")
    print("="*80)


if __name__ == "__main__":
    if not RAW_RECIPES.exists():
        print(f"❌ ERROR: {RAW_RECIPES} not found!")
        print(f"   Please ensure Food.com dataset is downloaded to {DATA_DIR / 'foodcom_dataset'}/")
        exit(1)
    
    if not INTERACTIONS_PATH.exists():
        print(f"⚠️  WARNING: {INTERACTIONS_PATH} not found!")
        print(f"   Ratings will be estimated without user interactions.")
    
    preprocess_recipes(RAW_RECIPES, INTERACTIONS_PATH, OUTPUT_PATH)
