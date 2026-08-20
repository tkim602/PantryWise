"""
Recipe Recommendation Pipeline (v2)
-----------------------------------

Implements the system outlined in SYSTEM_DESIGN.md using Open Food Facts
as the external source of Nutri-Score and Eco-Score labels.

Steps:
    1. Load Food.com recipes and Open Food Facts (OFF) products.
    2. Build aligned numeric feature matrices (nutrient macros per 100g).
    3. Train LightGBM regressors for:
         - Taste (Food.com avg_rating)
         - Nutrition (Nutri-Score from OFF)
         - Sustainability (Eco-Score from OFF)
    4. Generate predicted scores for every Food.com recipe and save them
       for downstream recommendation/evaluation.

Run:
    source venv/bin/activate
    python model2_lightGBM_v2/pipeline.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, kendalltau

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FOODCOM_PATH = DATA_DIR / "recipes_processed.csv"
OFF_PATH = DATA_DIR / "open_food_facts_datasets" / "openfoodfacts_cleaned.csv"

ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ARTIFACT_DIR / "models"
OUTPUT_DIR = ARTIFACT_DIR / "data"
PLOTS_DIR = ARTIFACT_DIR / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature mapping between Food.com and OFF columns
FEATURE_MAP = [
    ("calories", "energy-kcal_100g"),
    ("fat", "fat_100g"),
    ("sat_fat", "saturated-fat_100g"),
    ("carbs", "carbohydrates_100g"),
    ("sugar", "sugars_100g"),
    ("protein", "proteins_100g"),
    ("sodium", "sodium_100g"),
]
FEATURE_NAMES = [f for f, _ in FEATURE_MAP]
OFF_FEATURE_NAMES = [off for _, off in FEATURE_MAP]
RENAME_MAP = {off: feature for feature, off in FEATURE_MAP}

TASTE_EXTRA_FEATURES = ["minutes", "n_steps", "n_ingredients"]

MAX_NUTRI_SAMPLES = 200_000
MAX_ECO_SAMPLES = 150_000

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 40,
    "verbose": -1,
}


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #


def load_foodcom() -> pd.DataFrame:
    cols = [
        "id",
        "name",
        "minutes",
        "n_steps",
        "n_ingredients",
        "ingredient_text",
        "calories",
        "fat",
        "sat_fat",
        "carbs",
        "sugar",
        "protein",
        "sodium",
        "avg_rating",
    ]
    df = pd.read_csv(FOODCOM_PATH, usecols=cols)
    return df


def load_open_food_facts() -> pd.DataFrame:
    usecols = [
        "product_name",
        "nutriscore_score",
        "ecoscore_score",
        *OFF_FEATURE_NAMES,
    ]
    df = pd.read_csv(OFF_PATH, usecols=usecols)
    return df


def sample_label_dataframe(df: pd.DataFrame, label: str, max_samples: int) -> pd.DataFrame:
    mask = df[label].notna()
    subset = df.loc[mask].copy()
    if max_samples and len(subset) > max_samples:
        subset = subset.sample(n=max_samples, random_state=42)
    return subset.reset_index(drop=True)


def extract_off_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[OFF_FEATURE_NAMES].rename(columns=RENAME_MAP)
    return features


def extract_food_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_NAMES].copy()


def build_tfidf_features(df: pd.DataFrame, text_column: str, n_components: int = 100) -> np.ndarray:
    """
    Build TF-IDF features from text column and reduce with SVD.
    
    Args:
        df: DataFrame with text column
        text_column: Name of the text column (e.g., 'ingredient_text' or 'product_name')
        n_components: Number of SVD components (default: 100)
    
    Returns:
        numpy array of shape (n_samples, n_components)
    """
    print(f"\n  Building TF-IDF features from '{text_column}'...")
    
    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    texts = df[text_column].fillna('')
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Reduce dimensionality with SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_svd = svd.fit_transform(tfidf_matrix)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"    SVD reduced to {n_components} components (explained variance: {explained_var:.1%})")
    
    return tfidf_svd.astype(np.float32)


def build_feature_matrices(
    food_df: pd.DataFrame, nutri_df: pd.DataFrame, eco_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Create aligned macro feature matrices and shared fill values."""
    nutri_features = extract_off_features(nutri_df)
    eco_features = extract_off_features(eco_df)
    off_features_all = pd.concat([nutri_features, eco_features], ignore_index=True)
    medians = off_features_all.median()

    nutri_features = nutri_features.fillna(medians)
    eco_features = eco_features.fillna(medians)
    food_macros = extract_food_features(food_df).fillna(medians)

    return {
        "food": food_macros,
        "nutri": nutri_features,
        "eco": eco_features,
        "medians": medians,
    }


# --------------------------------------------------------------------------- #
# Model training utilities
# --------------------------------------------------------------------------- #


@dataclass
class ModelArtifacts:
    booster: lgb.Booster
    metrics: Dict[str, float]
    model_path: Path
    y_val: np.ndarray
    val_pred: np.ndarray


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        # 1. Root Mean Squared Error
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        
        # 2. Mean Absolute Error
        "mae": float(mean_absolute_error(y_true, y_pred)),
        
        # 3. R² Score (Coefficient of Determination)
        "r2": float(r2_score(y_true, y_pred)),
        
        # 4. Mean Absolute Percentage Error
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)) if np.all(y_true != 0) else 0.0,
        
        # 5. Max Error (worst prediction)
        "max_error": float(np.max(np.abs(y_true - y_pred))),
        
        # 6. Median Absolute Error
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        
        # 7. Explained Variance Score
        "explained_variance": float(1 - np.var(y_true - y_pred) / np.var(y_true)),
    }
    return metrics


def train_regressor(X: np.ndarray, y: np.ndarray, model_name: str) -> ModelArtifacts:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        valid_sets=[val_set],
        num_boost_round=4000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    val_pred = booster.predict(X_val)
    metrics = calculate_metrics(y_val, val_pred)
    
    model_path = MODEL_DIR / f"{model_name}.txt"
    booster.save_model(model_path)
    return ModelArtifacts(
        booster=booster, 
        metrics=metrics, 
        model_path=model_path,
        y_val=y_val,
        val_pred=val_pred
    )


def save_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Visualization utilities
# --------------------------------------------------------------------------- #


def plot_model_evaluation(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    score_range: tuple = None
) -> None:
    """
    Create comprehensive evaluation plots for a model.
    
    Generates 5 subplots:
      1. Actual vs Predicted scatter
      2. Residuals distribution
      3. Residuals vs Predicted
      4. Metrics bar chart
      5. Prediction error histogram
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} - Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    ax2.set_xlabel('Residuals (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals vs Predicted
    ax3 = axes[0, 2]
    ax3.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Predicted')
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Bar Chart
    ax4 = axes[1, 0]
    metric_names = ['RMSE', 'MAE', 'R²', 'MAPE', 'Median AE']
    metric_values = [
        metrics['rmse'],
        metrics['mae'],
        metrics['r2'],
        metrics['mape'],
        metrics['median_ae']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax4.barh(metric_names, metric_values, color=colors, edgecolor='black')
    ax4.set_xlabel('Value')
    ax4.set_title('Evaluation Metrics')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax4.text(val, i, f' {val:.3f}', va='center', fontweight='bold')
    
    # 5. Prediction Error Histogram
    ax5 = axes[1, 1]
    errors = np.abs(residuals)
    ax5.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax5.axvline(x=metrics['mae'], color='blue', linestyle='--', lw=2, label=f"MAE: {metrics['mae']:.3f}")
    ax5.axvline(x=metrics['median_ae'], color='green', linestyle='--', lw=2, label=f"Median AE: {metrics['median_ae']:.3f}")
    ax5.set_xlabel('Absolute Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Prediction Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    metrics_text = [
        f"{'Metric':<20} {'Value':>10}",
        "=" * 32,
        f"{'RMSE':<20} {metrics['rmse']:>10.4f}",
        f"{'MAE':<20} {metrics['mae']:>10.4f}",
        f"{'R² Score':<20} {metrics['r2']:>10.4f}",
        f"{'MAPE (%)':<20} {metrics['mape']*100:>10.2f}",
        f"{'Max Error':<20} {metrics['max_error']:>10.4f}",
        f"{'Median AE':<20} {metrics['median_ae']:>10.4f}",
        f"{'Explained Var':<20} {metrics['explained_variance']:>10.4f}",
        "=" * 32,
        f"{'Samples':<20} {len(y_true):>10,}",
    ]
    
    ax6.text(0.1, 0.9, '\n'.join(metrics_text), 
             fontfamily='monospace', fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = PLOTS_DIR / f"{model_name}_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Saved plot → {plot_path}")
    plt.close()


def plot_combined_metrics(all_metrics: Dict[str, Dict[str, float]]) -> None:
    """Create a combined comparison plot of all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Comparison - Key Metrics', fontsize=16, fontweight='bold')
    
    models = list(all_metrics.keys())
    
    # 1. RMSE Comparison
    ax1 = axes[0]
    rmse_values = [all_metrics[m]['rmse'] for m in models]
    bars1 = ax1.bar(models, rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Root Mean Squared Error')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. R² Comparison
    ax2 = axes[1]
    r2_values = [all_metrics[m]['r2'] for m in models]
    bars2 = ax2.bar(models, r2_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score (Coefficient of Determination)')
    ax2.axhline(y=0, color='red', linestyle='--', lw=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # 3. MAE Comparison
    ax3 = axes[2]
    mae_values = [all_metrics[m]['mae'] for m in models]
    bars3 = ax3.bar(models, mae_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
    ax3.set_ylabel('MAE')
    ax3.set_title('Mean Absolute Error')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, mae_values):
        ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comparison plot → {plot_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# Recommendation utilities
# --------------------------------------------------------------------------- #


def calculate_pantry_coverage(ingredient_text: str, user_pantry: Set[str]) -> float:
    """
    Calculate what percentage of recipe ingredients are in user's pantry.
    
    Args:
        ingredient_text: Comma-separated ingredient list (e.g., "chicken breast, pasta, olive oil")
        user_pantry: Set of ingredients user has (e.g., {'chicken', 'pasta', 'tomato'})
    
    Returns:
        Coverage ratio between 0.0 and 1.0
    """
    if not ingredient_text or pd.isna(ingredient_text):
        return 0.0
    
    # Parse ingredients (lowercase and split by comma)
    recipe_ingredients = [ing.strip().lower() for ing in str(ingredient_text).split(',')]
    
    if not recipe_ingredients:
        return 0.0
    
    # Count matches using substring matching
    matched = sum(
        1 for recipe_ing in recipe_ingredients
        if any(pantry_item.lower() in recipe_ing for pantry_item in user_pantry)
    )
    
    return matched / len(recipe_ingredients)


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize predicted scores to 0-1 scale for weighted combination.
    
    Normalization rules:
      - Taste: 1-5 stars → 0-1 (higher is better)
      - Nutri-Score: -15 to 40 → 1-0 (lower is better, so invert)
      - Eco-Score: 0-100 → 0-1 (higher is better)
    """
    df = df.copy()
    
    # Taste: (score - 1) / 4  →  [1,5] → [0,1]
    df['norm_taste'] = (df['pred_taste'] - 1.0) / 4.0
    
    # Nutrition: 1 - (score + 15) / 55  →  [-15,40] → [1,0]
    # Lower Nutri-Score = healthier, so invert
    df['norm_nutrition'] = 1.0 - ((df['pred_nutriscore'] + 15.0) / 55.0)
    
    # Sustainability: score / 100  →  [0,100] → [0,1]
    df['norm_sustainability'] = df['pred_ecoscore'] / 100.0
    
    # Clip to valid range
    df['norm_taste'] = df['norm_taste'].clip(0, 1)
    df['norm_nutrition'] = df['norm_nutrition'].clip(0, 1)
    df['norm_sustainability'] = df['norm_sustainability'].clip(0, 1)
    
    return df


def nutriscore_to_category(score: float) -> str:
    """Convert Nutri-Score points to letter grade (A/B/C/D/E) for general foods."""
    if score <= 2:
        return 'A'
    elif score <= 10:
        return 'B'
    elif score <= 18:
        return 'C'
    elif score <= 26:
        return 'D'
    else:
        return 'E'


def calculate_nutriscore_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate Nutri-Score classification accuracy metrics."""
    true_categories = [nutriscore_to_category(s) for s in y_true]
    pred_categories = [nutriscore_to_category(s) for s in y_pred]
    
    from sklearn.metrics import accuracy_score
    exact_accuracy = accuracy_score(true_categories, pred_categories)
    
    category_order = ['A', 'B', 'C', 'D', 'E']
    true_indices = [category_order.index(c) for c in true_categories]
    pred_indices = [category_order.index(c) for c in pred_categories]
    
    within_1 = sum(abs(t - p) <= 1 for t, p in zip(true_indices, pred_indices))
    within_1_accuracy = within_1 / len(true_categories)
    
    return {
        'exact_accuracy': exact_accuracy,
        'within_1_accuracy': within_1_accuracy,
        'true_categories': true_categories,
        'pred_categories': pred_categories
    }


def calculate_ecoscore_rank_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate Eco-Score ranking correlation metrics."""
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    kendall_corr, kendall_p = kendalltau(y_true, y_pred)
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'kendall_tau': kendall_corr,
        'kendall_pvalue': kendall_p
    }


def calculate_precision_at_k(recommended_df: pd.DataFrame, k: int, good_threshold: float = 4.5) -> float:
    """
    Calculate Precision@K: proportion of recommended items that are actually GOOD.
    """
    if len(recommended_df) == 0:
        return 0.0
    
    top_k = recommended_df.head(k)
    n_good = (top_k['avg_rating'] >= good_threshold).sum()
    return n_good / min(k, len(top_k))


def calculate_ndcg_at_k(recommended_df: pd.DataFrame, k: int) -> float:
    """
    Calculate nDCG@K: ranking quality metric considering position.
    """
    if len(recommended_df) == 0:
        return 0.0
    
    top_k = recommended_df.head(k)
    y_true = top_k['avg_rating'].values.reshape(1, -1)
    y_score = top_k['final_score'].values.reshape(1, -1)
    
    try:
        ndcg = ndcg_score(y_true, y_score, k=k)
        return ndcg
    except:
        return 0.0


def calculate_baseline_metrics(recipes_df: pd.DataFrame, user_pantry: Set[str], 
                               min_coverage: float, k: int, n_samples: int = 100) -> Dict[str, float]:
    """
    Calculate baseline metrics by averaging over random samples.
    """
    df = recipes_df.copy()
    df['pantry_coverage'] = df['ingredient_text'].apply(
        lambda x: calculate_pantry_coverage(x, user_pantry)
    )
    eligible = df[df['pantry_coverage'] >= min_coverage].copy()
    
    if len(eligible) < k:
        return {'nutriscore': 0, 'ecoscore': 0, 'avg_rating': 0}
    
    nutri_scores = []
    eco_scores = []
    ratings = []
    
    for _ in range(n_samples):
        sample = eligible.sample(n=min(k, len(eligible)), random_state=None)
        nutri_scores.append(sample['pred_nutriscore'].mean())
        eco_scores.append(sample['pred_ecoscore'].mean())
        ratings.append(sample['avg_rating'].mean())
    
    return {
        'nutriscore': np.mean(nutri_scores),
        'ecoscore': np.mean(eco_scores),
        'avg_rating': np.mean(ratings)
    }


def plot_recommendation_metrics(
    recommendations: pd.DataFrame,
    baseline: Dict[str, float],
    precision_at_5: float,
    precision_at_10: float,
    ndcg_at_5: float,
    ndcg_at_10: float
) -> None:
    """Create comprehensive visualization of recommendation quality metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Recommendation System Evaluation Metrics', fontsize=18, fontweight='bold')
    
    # 1. Precision@K
    ax1 = fig.add_subplot(gs[0, 0])
    k_values = [5, 10]
    precision_values = [precision_at_5, precision_at_10]
    bars1 = ax1.bar(k_values, precision_values, color=['#FF6B6B', '#4ECDC4'], 
                    edgecolor='black', width=2.5, alpha=0.8)
    ax1.set_xlabel('K (Top-K Recommendations)', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision@K (% of GOOD recipes with rating ≥ 4.5)', fontsize=13, fontweight='bold')
    ax1.set_xticks(k_values)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, precision_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. nDCG@K
    ax2 = fig.add_subplot(gs[0, 1])
    ndcg_values = [ndcg_at_5, ndcg_at_10]
    bars2 = ax2.bar(k_values, ndcg_values, color=['#45B7D1', '#FFA07A'], 
                    edgecolor='black', width=2.5, alpha=0.8)
    ax2.set_xlabel('K (Top-K Recommendations)', fontsize=12)
    ax2.set_ylabel('nDCG Score', fontsize=12)
    ax2.set_title('nDCG@K (Ranking Quality)', fontsize=13, fontweight='bold')
    ax2.set_xticks(k_values)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, ndcg_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Multi-Objective Scores (Our Method vs Baseline)
    ax3 = fig.add_subplot(gs[1, 0])
    
    our_nutriscore = recommendations['pred_nutriscore'].mean()
    our_ecoscore = recommendations['pred_ecoscore'].mean()
    our_rating = recommendations['avg_rating'].mean()
    
    categories = ['Nutri-Score\n(lower better)', 'Eco-Score\n(higher better)', 'Taste Rating\n(higher better)']
    our_values = [our_nutriscore, our_ecoscore, our_rating]
    baseline_values = [baseline['nutriscore'], baseline['ecoscore'], baseline['avg_rating']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars_baseline = ax3.bar(x - width/2, baseline_values, width, label='Random Baseline',
                           color='#95a5a6', edgecolor='black', alpha=0.7)
    bars_ours = ax3.bar(x + width/2, our_values, width, label='Our Method',
                       color='#27ae60', edgecolor='black', alpha=0.9)
    
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Multi-Objective Performance vs Baseline', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars_baseline:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars_ours:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Improvement Metrics
    ax4 = fig.add_subplot(gs[1, 1])
    
    nutri_improvement = baseline['nutriscore'] - our_nutriscore  # Lower is better
    eco_improvement = our_ecoscore - baseline['ecoscore']  # Higher is better
    rating_improvement = our_rating - baseline['avg_rating']  # Higher is better
    
    improvements = {
        'Nutrition\n(Nutri-Score ↓)': nutri_improvement,
        'Sustainability\n(Eco-Score ↑)': eco_improvement,
        'Taste\n(Rating ↑)': rating_improvement
    }
    
    colors_improvement = ['#27ae60' if v > 0 else '#e74c3c' for v in improvements.values()]
    bars4 = ax4.barh(list(improvements.keys()), list(improvements.values()),
                     color=colors_improvement, edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Improvement over Baseline', fontsize=12)
    ax4.set_title('Score Improvements (Our Method - Baseline)', fontsize=13, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars4, improvements.values()):
        width_bar = bar.get_width()
        label_x = width_bar + (0.1 if width_bar > 0 else -0.1)
        ha_align = 'left' if width_bar > 0 else 'right'
        ax4.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:+.2f} {"✅" if val > 0 else "❌"}',
                ha=ha_align, va='center', fontweight='bold', fontsize=11)
    
    # Save plot
    plot_path = PLOTS_DIR / "recommendation_metrics_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved recommendation metrics plot → {plot_path}")
    plt.close()


def recommend_recipes(
    recipes_df: pd.DataFrame,
    user_pantry: Set[str],
    weight_taste: float = 0.4,
    weight_nutrition: float = 0.3,
    weight_sustainability: float = 0.3,
    min_coverage: float = 0.8,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Recommend top-K recipes based on pantry coverage and weighted score.
    
    Args:
        recipes_df: DataFrame with predictions (pred_taste, pred_nutriscore, pred_ecoscore)
        user_pantry: Set of ingredients user has
        weight_taste: Weight for taste score (default: 0.4)
        weight_nutrition: Weight for nutrition score (default: 0.3)
        weight_sustainability: Weight for sustainability score (default: 0.3)
        min_coverage: Minimum pantry coverage required (default: 0.8 = 80%)
        top_k: Number of recipes to recommend (default: 10)
    
    Returns:
        DataFrame with top-K recommended recipes sorted by final_score
    """
    df = recipes_df.copy()
    
    # Step 1: Calculate pantry coverage for each recipe
    print(f"\n=== Filtering by pantry coverage (>= {min_coverage:.0%}) ===")
    df['pantry_coverage'] = df['ingredient_text'].apply(
        lambda x: calculate_pantry_coverage(x, user_pantry)
    )
    
    # Filter by minimum coverage
    df = df[df['pantry_coverage'] >= min_coverage].copy()
    print(f"Recipes with >= {min_coverage:.0%} coverage: {len(df):,}")
    
    if len(df) == 0:
        print("⚠️  No recipes meet the coverage threshold!")
        return pd.DataFrame()
    
    # Step 2: Normalize scores to 0-1 scale
    df = normalize_scores(df)
    
    # Step 3: Calculate weighted final score
    df['final_score'] = (
        weight_taste * df['norm_taste'] +
        weight_nutrition * df['norm_nutrition'] +
        weight_sustainability * df['norm_sustainability']
    )
    
    # Step 4: Sort by final score and return top-K
    df = df.sort_values('final_score', ascending=False).head(top_k)
    
    # Calculate missing ingredients for display
    def get_missing_ingredients(ingredient_text: str, user_pantry: Set[str]) -> List[str]:
        if not ingredient_text or pd.isna(ingredient_text):
            return []
        recipe_ingredients = [ing.strip().lower() for ing in str(ingredient_text).split(',')]
        missing = [
            ing for ing in recipe_ingredients
            if not any(pantry_item.lower() in ing for pantry_item in user_pantry)
        ]
        return missing
    
    df['missing_ingredients'] = df['ingredient_text'].apply(
        lambda x: get_missing_ingredients(x, user_pantry)
    )
    
    return df


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def main() -> None:
    print("=== Loading datasets ===")
    food_df = load_foodcom()
    off_df = load_open_food_facts()
    print(f"Food.com recipes: {len(food_df):,}")
    print(f"OFF products:     {len(off_df):,}")

    print("\n=== Sampling OFF labels ===")
    nutri_df = sample_label_dataframe(off_df, "nutriscore_score", MAX_NUTRI_SAMPLES)
    eco_df = sample_label_dataframe(off_df, "ecoscore_score", MAX_ECO_SAMPLES)
    print(f"Nutri-Score samples: {len(nutri_df):,}")
    print(f"Eco-Score samples:   {len(eco_df):,}")

    print("\n=== Building feature matrices ===")
    features = build_feature_matrices(food_df, nutri_df, eco_df)
    food_macro = features["food"].values.astype(np.float32)
    nutri_X = features["nutri"].values.astype(np.float32)
    eco_X = features["eco"].values.astype(np.float32)

    # Build TF-IDF features
    tfidf_food = build_tfidf_features(food_df, 'ingredient_text', n_components=100)
    tfidf_nutri = build_tfidf_features(nutri_df, 'product_name', n_components=100)
    tfidf_eco = build_tfidf_features(eco_df, 'product_name', n_components=100)

    # Taste features: macros + metadata + TF-IDF
    extra_feats = food_df[TASTE_EXTRA_FEATURES].fillna(0).values.astype(np.float32)
    taste_X = np.hstack([food_macro, extra_feats, tfidf_food])
    print(f"  Taste features shape: {taste_X.shape} (macros: 7, metadata: 3, TF-IDF: 100)")
    
    # Nutrition features: macros + TF-IDF
    nutri_X_enhanced = np.hstack([nutri_X, tfidf_nutri])
    print(f"  Nutrition features shape: {nutri_X_enhanced.shape} (macros: 7, TF-IDF: 100)")
    
    # Sustainability features: macros + TF-IDF
    eco_X_enhanced = np.hstack([eco_X, tfidf_eco])
    print(f"  Sustainability features shape: {eco_X_enhanced.shape} (macros: 7, TF-IDF: 100)")

    print("\n=== Training models ===")
    print("\n[1/3] Training Taste Model (with TF-IDF)...")
    taste_artifacts = train_regressor(taste_X, food_df["avg_rating"].values, "taste_model")
    print(f"  ✓ Taste metrics: RMSE={taste_artifacts.metrics['rmse']:.4f}, R²={taste_artifacts.metrics['r2']:.4f}")

    print("\n[2/3] Training Nutrition Model (with TF-IDF)...")
    y_nutri = nutri_df["nutriscore_score"].clip(-15, 40).values
    nutrition_artifacts = train_regressor(nutri_X_enhanced, y_nutri, "nutrition_model")
    print(f"  ✓ Nutrition metrics: RMSE={nutrition_artifacts.metrics['rmse']:.4f}, R²={nutrition_artifacts.metrics['r2']:.4f}")

    print("\n[3/3] Training Sustainability Model (with TF-IDF)...")
    y_eco = eco_df["ecoscore_score"].clip(0, 100).values
    sustainability_artifacts = train_regressor(eco_X_enhanced, y_eco, "sustainability_model")
    print(f"  ✓ Sustainability metrics: RMSE={sustainability_artifacts.metrics['rmse']:.4f}, R²={sustainability_artifacts.metrics['r2']:.4f}")

    # Save all metrics
    all_metrics = {
        "Taste": taste_artifacts.metrics,
        "Nutrition": nutrition_artifacts.metrics,
        "Sustainability": sustainability_artifacts.metrics,
    }
    save_json(all_metrics, OUTPUT_DIR / "training_metrics.json")
    
    # Generate evaluation plots
    print("\n=== Generating evaluation plots ===")
    plot_model_evaluation("Taste", taste_artifacts.y_val, taste_artifacts.val_pred, 
                         taste_artifacts.metrics, score_range=(1, 5))
    
    # Calculate Nutri-Score category metrics
    nutriscore_metrics = calculate_nutriscore_metrics(nutrition_artifacts.y_val, nutrition_artifacts.val_pred)
    all_metrics["Nutrition"]["nutriscore_exact_accuracy"] = nutriscore_metrics['exact_accuracy']
    all_metrics["Nutrition"]["nutriscore_within1_accuracy"] = nutriscore_metrics['within_1_accuracy']
    print(f"  ℹ️  Nutri-Score Category Accuracy: {nutriscore_metrics['exact_accuracy']:.1%}, Within-1: {nutriscore_metrics['within_1_accuracy']:.1%}")
    
    plot_model_evaluation("Nutrition", nutrition_artifacts.y_val, nutrition_artifacts.val_pred,
                         nutrition_artifacts.metrics, score_range=(-15, 40))
    
    # Calculate Eco-Score ranking metrics
    ecoscore_metrics = calculate_ecoscore_rank_metrics(sustainability_artifacts.y_val, sustainability_artifacts.val_pred)
    all_metrics["Sustainability"]["ecoscore_spearman"] = ecoscore_metrics['spearman_correlation']
    all_metrics["Sustainability"]["ecoscore_kendall"] = ecoscore_metrics['kendall_tau']
    print(f"  ℹ️  Eco-Score Spearman ρ: {ecoscore_metrics['spearman_correlation']:.4f}, Kendall τ: {ecoscore_metrics['kendall_tau']:.4f}")
    
    plot_model_evaluation("Sustainability", sustainability_artifacts.y_val, sustainability_artifacts.val_pred,
                         sustainability_artifacts.metrics, score_range=(0, 100))
    plot_combined_metrics(all_metrics)

    print("\n=== Generating Food.com predictions ===")
    # Taste: use full feature set (macros + metadata + TF-IDF)
    food_df["pred_taste"] = taste_artifacts.booster.predict(taste_X)
    
    # Nutrition & Sustainability: need to build enhanced features
    food_macro_enhanced = np.hstack([food_macro, tfidf_food])
    food_df["pred_nutriscore"] = nutrition_artifacts.booster.predict(food_macro_enhanced)
    food_df["pred_ecoscore"] = sustainability_artifacts.booster.predict(food_macro_enhanced)

    # Filter recipes with >= 5 ingredients (avoid side dishes and sauces)
    print(f"\n=== Filtering recipes (>= 5 ingredients) ===")
    print(f"Total recipes before filtering: {len(food_df):,}")
    food_df_filtered = food_df[food_df['n_ingredients'] >= 7].copy()
    print(f"Recipes with >= 5 ingredients: {len(food_df_filtered):,} ({len(food_df_filtered)/len(food_df)*100:.1f}%)")
    print(f"Filtered out (side dishes/sauces): {len(food_df) - len(food_df_filtered):,}")

    output_cols = [
        "id",
        "name",
        "avg_rating",
        "pred_taste",
        "pred_nutriscore",
        "pred_ecoscore",
        "minutes",
        "n_steps",
        "n_ingredients",
        "ingredient_text",
        "calories",
        "fat",
        "sat_fat",
        "carbs",
        "sugar",
        "protein",
        "sodium",
    ]
    food_df_filtered[output_cols].to_csv(
        OUTPUT_DIR / "recipes_with_predictions.csv", index=False
    )
    print(f"Saved predictions → {OUTPUT_DIR / 'recipes_with_predictions.csv'}")
    
    # Use filtered dataframe for recommendations
    food_df = food_df_filtered
    
    # --------------------------------------------------------------------------- #
    # RECOMMENDATION DEMO
    # --------------------------------------------------------------------------- #
    
    print("\n" + "="*80)
    print("RECIPE RECOMMENDATION DEMO")
    print("="*80)
    
    # Example user pantry (modify this to test different scenarios)
    user_pantry = {
        'chicken', 'pasta', 'tomato', 'olive oil', 'garlic', 'onion',
        'flour', 'sugar', 'salt', 'beef', 'basil', 'soy sauce',
        'black pepper', 'potato'
    }
    
    print(f"\n📦 User's Pantry ({len(user_pantry)} items):")
    print(f"   {', '.join(sorted(user_pantry))}")
    
    # Get recommendations with weighted scoring
    recommendations = recommend_recipes(
        recipes_df=food_df,
        user_pantry=user_pantry,
        weight_taste=0.4,          # 40% taste
        weight_nutrition=0.3,      # 30% nutrition
        weight_sustainability=0.3, # 30% sustainability
        min_coverage=0.8,          # 80% pantry coverage required
        top_k=10                   # Top 10 recipes
    )
    
    if len(recommendations) > 0:
        print(f"\n✅ Found {len(recommendations)} recommendations!")
        
        # Calculate ranking metrics
        print("\n" + "="*80)
        print("RANKING QUALITY METRICS")
        print("="*80)
        
        precision_at_5 = calculate_precision_at_k(recommendations, k=5, good_threshold=4.5)
        precision_at_10 = calculate_precision_at_k(recommendations, k=10, good_threshold=4.5)
        ndcg_at_5 = calculate_ndcg_at_k(recommendations, k=5)
        ndcg_at_10 = calculate_ndcg_at_k(recommendations, k=10)
        
        print(f"\n📊 Precision@K (% of GOOD recipes with rating >= 4.5):")
        print(f"   ├─ Precision@5:  {precision_at_5:.1%}")
        print(f"   └─ Precision@10: {precision_at_10:.1%}")
        
        print(f"\n📈 nDCG@K (ranking quality, higher = better ratings at top):")
        print(f"   ├─ nDCG@5:  {ndcg_at_5:.4f}")
        print(f"   └─ nDCG@10: {ndcg_at_10:.4f}")
        
        # Calculate baseline comparison
        print("\n" + "="*80)
        print("MULTI-OBJECTIVE EFFECTIVENESS (vs Random Baseline)")
        print("="*80)
        
        print("\n⏳ Calculating baseline metrics (100 random samples)...")
        baseline = calculate_baseline_metrics(
            recipes_df=food_df,
            user_pantry=user_pantry,
            min_coverage=0.8,
            k=10,
            n_samples=100
        )
        
        # Our method's average scores
        our_nutriscore = recommendations['pred_nutriscore'].mean()
        our_ecoscore = recommendations['pred_ecoscore'].mean()
        our_rating = recommendations['avg_rating'].mean()
        
        nutri_improvement = baseline['nutriscore'] - our_nutriscore  # Lower is better
        eco_improvement = our_ecoscore - baseline['ecoscore']  # Higher is better
        rating_improvement = our_rating - baseline['avg_rating']  # Higher is better
        
        print(f"\n🥗 Nutrition (Nutri-Score, lower = healthier):")
        print(f"   ├─ Random Baseline:  {baseline['nutriscore']:.2f}")
        print(f"   ├─ Our Method:       {our_nutriscore:.2f}")
        print(f"   └─ Improvement:      {nutri_improvement:+.2f} {'✅' if nutri_improvement > 0 else '⚠️'}")
        
        print(f"\n🌱 Sustainability (Eco-Score, higher = better):")
        print(f"   ├─ Random Baseline:  {baseline['ecoscore']:.2f}")
        print(f"   ├─ Our Method:       {our_ecoscore:.2f}")
        print(f"   └─ Improvement:      {eco_improvement:+.2f} {'✅' if eco_improvement > 0 else '⚠️'}")
        
        print(f"\n😋 Taste (Average Rating, higher = better):")
        print(f"   ├─ Random Baseline:  {baseline['avg_rating']:.2f}")
        print(f"   ├─ Our Method:       {our_rating:.2f}")
        print(f"   └─ Improvement:      {rating_improvement:+.2f} {'✅' if rating_improvement > 0 else '⚠️'}")
        
        # Generate recommendation metrics visualization
        print("\n=== Generating recommendation metrics plots ===")
        plot_recommendation_metrics(
            recommendations=recommendations,
            baseline=baseline,
            precision_at_5=precision_at_5,
            precision_at_10=precision_at_10,
            ndcg_at_5=ndcg_at_5,
            ndcg_at_10=ndcg_at_10
        )
        
        print("\n" + "="*80)
        print("TOP RECOMMENDATIONS")
        print("="*80)
        
        for rank, row in enumerate(recommendations.itertuples(), 1):
            print(f"\n🏆 Rank #{rank}: {row.name}")
            print(f"   Final Score: {row.final_score:.3f}")
            print(f"   ├─ 😋 Taste:          {row.pred_taste:.2f}/5.0 (actual: {row.avg_rating:.2f}⭐)")
            print(f"   ├─ 🥗 Nutrition:      {row.pred_nutriscore:.1f} (Nutri-Score)")
            print(f"   ├─ 🌱 Sustainability: {row.pred_ecoscore:.1f}/100 (Eco-Score)")
            print(f"   ├─ 📦 Pantry Coverage: {row.pantry_coverage:.1%}")
            print(f"   ├─ ⏱️  Cook Time: {row.minutes:.0f} min")
            print(f"   ├─ 📝 Steps: {row.n_steps:.0f}")
            print(f"   └─ 🛒 Missing: {len(row.missing_ingredients)} items")
            if row.missing_ingredients:
                missing_str = ', '.join(row.missing_ingredients[:3])
                if len(row.missing_ingredients) > 3:
                    missing_str += f", ... (+{len(row.missing_ingredients)-3} more)"
                print(f"      → {missing_str}")
        
        # Save recommendations
        output_cols_rec = [
            "id", "name", "final_score", 
            "pred_taste", "pred_nutriscore", "pred_ecoscore",
            "norm_taste", "norm_nutrition", "norm_sustainability",
            "pantry_coverage", "minutes", "n_steps", "n_ingredients",
            "missing_ingredients", "ingredient_text"
        ]
        recommendations[output_cols_rec].to_csv(
            OUTPUT_DIR / "top_recommendations.csv", index=False
        )
        print(f"\n💾 Saved recommendations → {OUTPUT_DIR / 'top_recommendations.csv'}")
    else:
        print("\n⚠️  No recipes meet the criteria. Try lowering min_coverage or expanding your pantry.")
    
    print("\n" + "="*80)
    print("✅ Pipeline complete!")
    print("="*80)


if __name__ == "__main__":
    main()
