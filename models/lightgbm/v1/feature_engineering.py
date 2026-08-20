"""
Feature Engineering for Multi-Task Learning
Creates task-specific features and fits TF-IDF vectorizer
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def create_feature_sets(data_path='../data/recipes_processed.csv', output_dir='../data'):
    """Create 3 feature sets for multi-task learning"""
    
    print("="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    
    # Load processed data
    print("\n[1/5] Loading processed data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} recipes")
    required_cols = [
        'avg_rating', 'nutrition_score', 'sustainability_score',
        'ingredient_text', 'n_ingredients', 'minutes', 'n_steps'
    ]
    before = len(df)
    df = df.dropna(subset=required_cols)
    print(f"  Dropped {before - len(df)} rows with missing critical fields -> {len(df)}")
    
    # Step 1: Fit TF-IDF vectorizer
    print("\n[2/5] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=4000,
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform(df['ingredient_text'].fillna(''))
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Reduce TF-IDF with SVD to create dense signals for diet/nutrition classifiers
    print("\n[2b/5] Reducing TF-IDF with TruncatedSVD (50 components)...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    tfidf_svd = svd.fit_transform(tfidf_matrix)
    svd_cols = [f'tfidf_svd_{i}' for i in range(tfidf_svd.shape[1])]
    for i, col in enumerate(svd_cols):
        df[col] = tfidf_svd[:, i]
    
    # Calculate IDF values for avg_ingredient_commonality
    feature_names = vectorizer.get_feature_names_out()
    idf_values = dict(zip(feature_names, vectorizer.idf_))
    
    # Calculate avg IDF for each recipe (lower IDF = more common ingredients)
    print("\n[3/5] Calculating ingredient commonality...")
    def calc_avg_commonality(text):
        words = text.split()
        if len(words) == 0:
            return 5.0  # Neutral
        idfs = [idf_values.get(word, 10.0) for word in words]
        return np.mean(idfs)
    
    df['avg_ingredient_commonality'] = df['ingredient_text'].fillna('').apply(calc_avg_commonality)
    print(f"  Commonality - mean: {df['avg_ingredient_commonality'].mean():.3f}, "
          f"std: {df['avg_ingredient_commonality'].std():.3f}")
    
    # Step 2: Create feature sets
    print("\n[4/5] Creating task-specific features...")
    
    # Define target-specific features
    taste_features = ['n_ingredients', 'minutes', 'n_steps', 'avg_ingredient_commonality']
    # Nutrition: use density-style ratios so model must learn patterns, not copy formula
    df['protein_density'] = df['protein'] / (df['calories'] + 1e-6)
    df['sugar_density'] = df['sugar'] / (df['calories'] + 1e-6)
    df['sodium_density'] = df['sodium'] / (df['calories'] + 1e-6)
    nutrition_features = [
        'protein_density', 'sugar_density', 'sodium_density',
        'fat', 'sat_fat', 'carbs', 'calories', 'n_ingredients',
        'avg_ingredient_commonality', 'plant_ratio', 'meat_ratio'
    ] + svd_cols  # avoid leakage by excluding diet flags as inputs
    
    # Sustainability: remove direct carbon feature to avoid deterministic mapping
    sustainability_features = ['protein', 'fat', 'n_ingredients', 'avg_ingredient_commonality']
    
    # Diet: CRITICAL - Remove plant_ratio/meat_ratio to avoid label leakage!
    # These ratios are DETERMINISTIC with vegan label (meat_ratio=0 ⟺ vegan=1)
    # Model must learn from TF-IDF ingredient embeddings (real ML patterns)
    diet_features = [
        'protein_density', 'sugar_density', 'sodium_density',
        'fat', 'sat_fat', 'carbs', 'calories', 'n_ingredients',
        'avg_ingredient_commonality'
        # REMOVED: 'plant_ratio', 'meat_ratio' - too deterministic!
    ] + svd_cols  # TF-IDF embeddings capture 'chicken', 'tofu', 'beef' semantics
    y_diet = df[['flag_vegan', 'flag_vegetarian', 'flag_gluten_free', 'flag_low_sodium', 'flag_low_sugar', 'flag_healthy']].values
    
    # Extract feature matrices
    X_taste = df[taste_features].values
    X_nutrition = df[nutrition_features].fillna(0).values
    X_sustainability = df[sustainability_features].fillna(0).values
    X_diet = df[diet_features].fillna(0).values
    
    # Extract labels
    y_taste = df['avg_rating'].values
    y_nutrition = df['nutrition_score'].values
    y_sustainability = df['sustainability_score'].values
    # Diet labels already assembled above
    
    print(f"\n  Feature dimensions:")
    print(f"    Taste: {X_taste.shape} -> {y_taste.shape}")
    print(f"    Nutrition: {X_nutrition.shape} -> {y_nutrition.shape}")
    print(f"    Sustainability: {X_sustainability.shape} -> {y_sustainability.shape}")
    print(f"    Diet labels: {y_diet.shape}")
    
    # Step 3: Save everything
    print("\n[5/5] Saving features and vectorizer...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save vectorizer
    with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved: {output_dir}/tfidf_vectorizer.pkl")
    
    # Save enhanced data with features
    df.to_csv(f'{output_dir}/recipes_with_features.csv', index=False)
    print(f"  Saved: {output_dir}/recipes_with_features.csv")
    
    # Save feature matrices as numpy arrays
    np.save(f'{output_dir}/X_taste.npy', X_taste)
    np.save(f'{output_dir}/X_nutrition.npy', X_nutrition)
    np.save(f'{output_dir}/X_sustainability.npy', X_sustainability)
    np.save(f'{output_dir}/X_diet.npy', X_diet)
    np.save(f'{output_dir}/y_taste.npy', y_taste)
    np.save(f'{output_dir}/y_nutrition.npy', y_nutrition)
    np.save(f'{output_dir}/y_sustainability.npy', y_sustainability)
    np.save(f'{output_dir}/y_diet.npy', y_diet)
    print(f"  Saved: 7 numpy arrays (X_*.npy, y_*.npy)")
    
    # Statistics
    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    print(f"\nTaste features: {taste_features}")
    print(f"Nutrition features: {nutrition_features}")
    print(f"Sustainability features: {sustainability_features}")
    print(f"Diet labels: ['flag_vegan','flag_vegetarian','flag_gluten_free','flag_low_sodium','flag_low_sugar','flag_healthy']")
    print(f"TF-IDF SVD features: {len(svd_cols)} components")
    
    print(f"\nLabel distributions:")
    print(f"  Taste (avg_rating): {y_taste.mean():.3f} ± {y_taste.std():.3f}")
    print(f"  Nutrition (score): {y_nutrition.mean():.3f} ± {y_nutrition.std():.3f}")
    print(f"  Sustainability (score): {y_sustainability.mean():.3f} ± {y_sustainability.std():.3f}")
    
    return {
        'X_taste': X_taste,
        'X_nutrition': X_nutrition,
        'X_sustainability': X_sustainability,
        'y_taste': y_taste,
        'y_nutrition': y_nutrition,
        'y_sustainability': y_sustainability,
        'vectorizer': vectorizer,
        'data': df
    }

if __name__ == '__main__':
    results = create_feature_sets()
    print("\nFeature engineering complete!")
