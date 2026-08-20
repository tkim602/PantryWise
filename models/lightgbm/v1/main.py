"""
Main Script: Complete Multi-Task Recipe Recommendation System
Run this to execute the full pipeline from data preprocessing to recommendations
"""

import sys
from pathlib import Path

def main():
    """Execute full pipeline"""
    
    print("\n" + "="*80)
    print(" " * 15 + "MULTI-TASK RECIPE RECOMMENDATION SYSTEM")
    print(" " * 20 + "Sustainability & Nutrition Aware")
    print("="*80)
    
    # Check if models exist
    model_dir = Path('./models')
    data_dir = Path('../data')
    
    models_exist = (
        (model_dir / 'model_taste.txt').exists() and
        (model_dir / 'model_nutrition.txt').exists() and
        (model_dir / 'model_sustainability.txt').exists()
    )
    
    data_exists = (data_dir / 'recipes_with_features.csv').exists()
    
    if not models_exist or not data_exists:
        print("\n⚠️  Models or data not found. Running full pipeline...\n")
        
        # Step 1: Data preprocessing
        print("\n" + "="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        from data_preprocessing import preprocess_foodcom_data
        preprocess_foodcom_data()
        
        # Step 2: Feature engineering
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        from feature_engineering import create_feature_sets
        create_feature_sets()
        
        # Step 3: Train models
        print("\n" + "="*80)
        print("STEP 3: MULTI-TASK MODEL TRAINING")
        print("="*80)
        from train_multitask import train_multitask_models
        train_multitask_models()
    else:
        print("\n✓ Models and data found. Skipping training.\n")
    
    # Step 4: Recommendations
    print("\n" + "="*80)
    print("STEP 4: RECIPE RECOMMENDATIONS")
    print("="*80)
    
    from recommend import load_recommender
    
    # Load recommender
    recommender = load_recommender()
    
    # Example pantry items
    pantry_items = [
        # Proteins
        'chicken breast', 'ground beef', 'salmon', 'eggs',
        
        # Vegetables
        'onion', 'garlic', 'tomato', 'bell pepper', 'carrot', 
        'broccoli', 'spinach', 'mushroom', 'zucchini',
        
        # Staples
        'olive oil', 'butter', 'salt', 'pepper', 'soy sauce',
        'rice', 'pasta', 'flour', 'sugar',
        
        # Dairy
        'milk', 'cheddar cheese', 'parmesan cheese', 'cream',
        
        # Herbs & Spices
        'basil', 'oregano', 'paprika', 'cumin', 'ginger',
        
        # Other
        'lemon', 'lime', 'vinegar', 'chicken broth', 'tomato sauce'
    ]
    
    print(f"\n📦 PANTRY INVENTORY ({len(pantry_items)} items):")
    print("-" * 80)
    for i in range(0, len(pantry_items), 5):
        print("  " + ", ".join(pantry_items[i:i+5]))
    
    # ========== SCENARIO 1: MAIN DISH ==========
    print("\n\n" + "="*80)
    print("SCENARIO 1: MAIN DISH RECOMMENDATIONS")
    print("  Coverage threshold: 70%")
    print("  Weights: 40% Taste + 30% Nutrition + 30% Sustainability")
    print("="*80)
    
    recommendations_main = recommender.recommend(
        pantry_items=pantry_items,
        intent='main',
        coverage_threshold=0.7,
        top_k=10,
        weights={'taste': 0.4, 'nutrition': 0.3, 'sustainability': 0.3, 'diet': 0.0}
    )
    
    # ========== SCENARIO 2: SIDE DISH ==========
    print("\n\n" + "="*80)
    print("SCENARIO 2: SIDE DISH RECOMMENDATIONS")
    print("  Coverage threshold: 70%")
    print("  Weights: 40% Taste + 30% Nutrition + 30% Sustainability")
    print("="*80)
    
    recommendations_side = recommender.recommend(
        pantry_items=pantry_items,
        intent='side',
        coverage_threshold=0.7,
        top_k=10,
        weights={'taste': 0.4, 'nutrition': 0.3, 'sustainability': 0.3, 'diet': 0.0}
    )
    
    # ========== SCENARIO 3: NUTRITION-FOCUSED ==========
    print("\n\n" + "="*80)
    print("SCENARIO 3: NUTRITION-FOCUSED MAIN DISH")
    print("  Coverage threshold: 70%")
    print("  Weights: 20% Taste + 60% Nutrition + 20% Sustainability")
    print("="*80)
    
    recommendations_nutrition = recommender.recommend(
        pantry_items=pantry_items,
        intent='main',
        coverage_threshold=0.7,
        top_k=5,
        weights={'taste': 0.2, 'nutrition': 0.6, 'sustainability': 0.2, 'diet': 0.0}
    )
    
    # ========== SCENARIO 4: SUSTAINABILITY-FOCUSED ==========
    print("\n\n" + "="*80)
    print("SCENARIO 4: SUSTAINABILITY-FOCUSED MAIN DISH")
    print("  Coverage threshold: 70%")
    print("  Weights: 20% Taste + 20% Nutrition + 60% Sustainability")
    print("="*80)
    
    recommendations_sustainability = recommender.recommend(
        pantry_items=pantry_items,
        intent='main',
        coverage_threshold=0.7,
        top_k=5,
        weights={'taste': 0.2, 'nutrition': 0.2, 'sustainability': 0.6, 'diet': 0.0}
    )
    
    # ========== SCENARIO 5: DIET-FOCUSED (VEGAN) ==========
    print("\n\n" + "="*80)
    print("SCENARIO 5: VEGAN-FOCUSED MAIN DISH")
    print("  Coverage threshold: 70%")
    print("  Weights: 30% Taste + 20% Nutrition + 20% Sustainability + 30% Diet")
    print("  Diet: Vegan (soft ranking, no hard filter)")
    print("="*80)
    
    recommendations_vegan_soft = recommender.recommend(
        pantry_items=pantry_items,
        intent='main',
        coverage_threshold=0.7,
        top_k=5,
        weights={'taste': 0.3, 'nutrition': 0.2, 'sustainability': 0.2, 'diet': 0.3},
        diet_preference='vegan',
        diet_threshold=0.0  # No hard filter, just use diet score in ranking
    )
    
    # ========== SCENARIO 6: STRICT VEGAN (HARD FILTER) ==========
    print("\n\n" + "="*80)
    print("SCENARIO 6: STRICT VEGAN MAIN DISH (Hard Filter)")
    print("  Coverage threshold: 70%")
    print("  Weights: 40% Taste + 30% Nutrition + 20% Sustainability + 10% Diet")
    print("  Diet: Vegan (probability >= 0.6)")
    print("="*80)
    
    recommendations_vegan_strict = recommender.recommend(
        pantry_items=pantry_items,
        intent='main',
        coverage_threshold=0.7,
        top_k=5,
        weights={'taste': 0.4, 'nutrition': 0.3, 'sustainability': 0.2, 'diet': 0.1},
        diet_preference='vegan',
        diet_threshold=0.6  # Balanced: Precision ~95%, Recall ~70%
    )
    
    # Final summary
    print("\n\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\n✓ System successfully combines:")
    print("  1. Real user taste feedback (avg_rating from 1M+ reviews)")
    print("  2. Calculated nutrition quality (protein, fiber, sodium, sugar)")
    print("  3. Calculated sustainability (plant-based ratio, carbon footprint)")
    print("  4. Diet preferences (vegan/vegetarian/gluten-free/low-sodium/low-sugar/healthy)")
    print("\n✓ Multi-task learning ensures:")
    print("  - Each objective is learned independently")
    print("  - No overfitting to synthetic labels")
    print("  - Flexible weighting for different user preferences")
    print("\n✓ Diet integration:")
    print("  - Soft ranking: Set diet weight > 0, threshold = 0 (ranks by diet probability)")
    print("  - Hard filter: Set diet threshold > 0 (removes recipes below threshold)")
    print("  - Combined: Use both weight + threshold for balanced recommendations")
    print("\n✓ 4-stage pipeline:")
    print("  Stage 1: TF-IDF retrieval + 70% pantry coverage")
    print("  Stage 2: Intent filtering (main vs side dish)")
    print("  Stage 3: ML prediction (3 independent scores)")
    print("  Stage 4: Multi-objective ranking with custom weights")
    print("\n" + "="*80)
    print("🎉 Ready for deployment!")
    print("="*80 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
