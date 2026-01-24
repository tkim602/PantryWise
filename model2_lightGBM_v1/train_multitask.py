"""
Multi-Task Learning Training
Trains 3 independent LightGBM models for taste, nutrition, sustainability
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import pickle

def train_multitask_models(data_dir='../data', model_dir='./models'):
    """Train 3 independent LightGBM models"""
    
    print("="*70)
    print("MULTI-TASK MODEL TRAINING")
    print("="*70)
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load features
    print("\n[1/4] Loading features and labels...")
    X_taste = np.load(f'{data_dir}/X_taste.npy')
    X_nutrition = np.load(f'{data_dir}/X_nutrition.npy')
    X_sustainability = np.load(f'{data_dir}/X_sustainability.npy')
    X_diet = np.load(f'{data_dir}/X_diet.npy')
    y_taste = np.load(f'{data_dir}/y_taste.npy')
    y_nutrition = np.load(f'{data_dir}/y_nutrition.npy')
    y_sustainability = np.load(f'{data_dir}/y_sustainability.npy')
    y_diet = np.load(f'{data_dir}/y_diet.npy')
    
    print(f"  Taste: {X_taste.shape} -> {y_taste.shape}")
    print(f"  Nutrition: {X_nutrition.shape} -> {y_nutrition.shape}")
    print(f"  Sustainability: {X_sustainability.shape} -> {y_sustainability.shape}")
    print(f"  Diet: {X_diet.shape} -> {y_diet.shape}")
    
    # Step 2: Filter noisy labels (-1) from diet data
    print("\n[2/4] Filtering noisy labels from diet data...")
    
    # Count noisy labels per diet class
    diet_labels = ['vegan', 'vegetarian', 'gluten_free', 'low_sodium', 'low_sugar', 'healthy']
    print(f"  Original diet samples: {len(y_diet)}")
    for idx, label_name in enumerate(diet_labels):
        n_noisy = np.sum(y_diet[:, idx] == -1)
        n_positive = np.sum(y_diet[:, idx] == 1)
        n_negative = np.sum(y_diet[:, idx] == 0)
        print(f"    {label_name}: {n_positive} positive, {n_negative} negative, {n_noisy} noisy (-1)")
    
    # Filter: Keep only rows where ALL diet labels are not -1
    clean_mask = (y_diet != -1).all(axis=1)
    X_diet_clean = X_diet[clean_mask]
    y_diet_clean = y_diet[clean_mask]
    
    n_filtered = len(y_diet) - len(y_diet_clean)
    print(f"  Filtered {n_filtered} noisy samples ({n_filtered/len(y_diet)*100:.1f}%)")
    print(f"  Clean diet samples: {len(y_diet_clean)}")
    
    # Step 3: Train/test split
    print("\n[3/4] Creating train/test splits (80/20)...")
    
    X_taste_train, X_taste_test, y_taste_train, y_taste_test = train_test_split(
        X_taste, y_taste, test_size=0.2, random_state=42
    )
    X_nutr_train, X_nutr_test, y_nutr_train, y_nutr_test = train_test_split(
        X_nutrition, y_nutrition, test_size=0.2, random_state=42
    )
    X_sust_train, X_sust_test, y_sust_train, y_sust_test = train_test_split(
        X_sustainability, y_sustainability, test_size=0.2, random_state=42
    )
    # Diet multi-label: Use cleaned data
    X_diet_train, X_diet_test, y_diet_train, y_diet_test = train_test_split(
        X_diet_clean, y_diet_clean, test_size=0.2, random_state=42
    )
    
    print(f"  Train size: {len(X_taste_train)}, Test size: {len(X_taste_test)}")
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 20  # Prevent overfitting
    }
    
    # Step 4: Train models
    print("\n[4/5] Training 4 models...\n")
    
    results = {}
    
    # ========== MODEL 1: TASTE ==========
    print("-" * 70)
    print("MODEL 1: TASTE (predict avg_rating)")
    print("-" * 70)
    
    train_data_taste = lgb.Dataset(X_taste_train, label=y_taste_train)
    val_data_taste = lgb.Dataset(X_taste_test, label=y_taste_test, reference=train_data_taste)
    
    model_taste = lgb.train(
        params,
        train_data_taste,
        num_boost_round=1000,  # Increased from 500
        valid_sets=[val_data_taste],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]  # Increased patience
    )
    
    y_pred_taste = model_taste.predict(X_taste_test)
    rmse_taste = np.sqrt(mean_squared_error(y_taste_test, y_pred_taste))
    mae_taste = mean_absolute_error(y_taste_test, y_pred_taste)
    
    print(f"\n✓ Taste Model Results:")
    print(f"  RMSE: {rmse_taste:.4f}")
    print(f"  MAE: {mae_taste:.4f}")
    print(f"  Feature importance:")
    feature_names_taste = ['n_ingredients', 'minutes', 'n_steps', 'avg_ingredient_commonality']
    for name, importance in zip(feature_names_taste, model_taste.feature_importance()):
        print(f"    {name}: {importance}")
    
    results['taste'] = {
        'model': model_taste,
        'rmse': rmse_taste,
        'mae': mae_taste,
        'y_test': y_taste_test,
        'y_pred': y_pred_taste
    }
    
    # ========== MODEL 2: NUTRITION ==========
    print("\n" + "-" * 70)
    print("MODEL 2: NUTRITION (predict nutrition_score)")
    print("-" * 70)
    
    train_data_nutr = lgb.Dataset(X_nutr_train, label=y_nutr_train)
    val_data_nutr = lgb.Dataset(X_nutr_test, label=y_nutr_test, reference=train_data_nutr)
    
    model_nutrition = lgb.train(
        params,
        train_data_nutr,
        num_boost_round=500,
        valid_sets=[val_data_nutr],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    y_pred_nutr = model_nutrition.predict(X_nutr_test)
    rmse_nutr = np.sqrt(mean_squared_error(y_nutr_test, y_pred_nutr))
    mae_nutr = mean_absolute_error(y_nutr_test, y_pred_nutr)
    
    print(f"\n✓ Nutrition Model Results:")
    print(f"  RMSE: {rmse_nutr:.4f}")
    print(f"  MAE: {mae_nutr:.4f}")
    print(f"  Feature importance:")
    feature_names_nutr = [
        'protein_density', 'sugar_density', 'sodium_density',
        'fat', 'sat_fat', 'carbs', 'calories', 'n_ingredients', 'avg_ingredient_commonality',
        'plant_ratio', 'meat_ratio'
    ] + [f'tfidf_svd_{i}' for i in range(X_nutrition.shape[1] - 11)]
    for name, importance in zip(feature_names_nutr, model_nutrition.feature_importance()):
        print(f"    {name}: {importance}")
    
    results['nutrition'] = {
        'model': model_nutrition,
        'rmse': rmse_nutr,
        'mae': mae_nutr,
        'y_test': y_nutr_test,
        'y_pred': y_pred_nutr
    }
    
    # ========== MODEL 3: SUSTAINABILITY ==========
    print("\n" + "-" * 70)
    print("MODEL 3: SUSTAINABILITY (predict sustainability_score)")
    print("-" * 70)
    
    train_data_sust = lgb.Dataset(X_sust_train, label=y_sust_train)
    val_data_sust = lgb.Dataset(X_sust_test, label=y_sust_test, reference=train_data_sust)
    
    model_sustainability = lgb.train(
        params,
        train_data_sust,
        num_boost_round=500,
        valid_sets=[val_data_sust],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    y_pred_sust = model_sustainability.predict(X_sust_test)
    rmse_sust = np.sqrt(mean_squared_error(y_sust_test, y_pred_sust))
    mae_sust = mean_absolute_error(y_sust_test, y_pred_sust)
    
    print(f"\n✓ Sustainability Model Results:")
    print(f"  RMSE: {rmse_sust:.4f}")
    print(f"  MAE: {mae_sust:.4f}")
    print(f"  Feature importance:")
    feature_names_sust = ['protein', 'fat', 'n_ingredients', 'avg_ingredient_commonality']
    for name, importance in zip(feature_names_sust, model_sustainability.feature_importance()):
        print(f"    {name}: {importance}")
    
    results['sustainability'] = {
        'model': model_sustainability,
        'rmse': rmse_sust,
        'mae': mae_sust,
        'y_test': y_sust_test,
        'y_pred': y_pred_sust
    }
    
    # ========== MODEL 4: DIET MULTI-LABEL ==========
    print("\n" + "-" * 70)
    print("MODEL 4: DIET (multi-label: vegan/veg/gluten-free/low-sodium/low-sugar)")
    print("-" * 70)
    
    diet_models = []
    diet_metrics = []
    diet_labels = ['vegan', 'vegetarian', 'gluten_free', 'low_sodium', 'low_sugar', 'healthy']
    for idx, label in enumerate(diet_labels):
        train_ds = lgb.Dataset(X_diet_train, label=y_diet_train[:, idx])
        val_ds = lgb.Dataset(X_diet_test, label=y_diet_test[:, idx], reference=train_ds)
        model = lgb.train(
            {**params, 'objective': 'binary', 'metric': 'binary_logloss'},
            train_ds,
            num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        diet_models.append(model)
        pred = model.predict(X_diet_test)
        # Accuracy at multiple thresholds
        acc_05 = np.mean(((pred >= 0.5).astype(int)) == y_diet_test[:, idx])
        acc_08 = np.mean(((pred >= 0.8).astype(int)) == y_diet_test[:, idx])
        diet_metrics.append(acc_05)
        print(f"  {label} acc@0.5: {acc_05:.3f}, acc@0.8: {acc_08:.3f}")
    
    results['diet'] = {'models': diet_models, 'acc': diet_metrics, 'y_test': y_diet_test}
    
    # Step 5: Save models
    print("\n[5/5] Saving models...")
    model_taste.save_model(f'{model_dir}/model_taste.txt')
    model_nutrition.save_model(f'{model_dir}/model_nutrition.txt')
    model_sustainability.save_model(f'{model_dir}/model_sustainability.txt')
    for model, name in zip(diet_models, ['vegan', 'vegetarian', 'gluten_free', 'low_sodium', 'low_sugar', 'healthy']):
        model.save_model(f'{model_dir}/diet_{name}.txt')
    print(f"  Saved 3 models to: {model_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"✓ Taste Model:          RMSE={rmse_taste:.4f}, MAE={mae_taste:.4f}")
    print(f"✓ Nutrition Model:      RMSE={rmse_nutr:.4f}, MAE={mae_nutr:.4f}")
    print(f"✓ Sustainability Model: RMSE={rmse_sust:.4f}, MAE={mae_sust:.4f}")
    print(f"\nDiet Models (threshold @0.5 for reference, @0.8 for strict filtering):")
    for name, acc in zip(diet_labels, diet_metrics):
        print(f"  {name:15s} Acc@0.5={acc:.3f}")
    print(f"\nAll models saved to: {model_dir}/")
    
    return results

if __name__ == '__main__':
    results = train_multitask_models()
    print("\nTraining complete!")
