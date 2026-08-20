"""
Recipe Recommendation Pipeline (v2 - Binary Classification for Taste)
-----------------------------------------------------------------------

Same as pipeline.py but uses binary classification for Taste as a quality filter.

Classes:
  - NOT_RECOMMENDED (0): rating < 4.5  (Below excellent, not worth recommending)
  - GOOD (1):            rating >= 4.5 (Excellent, worth recommending)

This approach treats taste prediction as a quality gate rather than precise scoring,
since user ratings are subjective and poorly correlated with ingredient composition.

Run:
    python model2_lightGBM_v2/pipeline_classification.py
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
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
PLOTS_DIR = ARTIFACT_DIR / "plots_classification"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature mapping
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

# LightGBM params for regression (nutrition/sustainability)
LGB_PARAMS_REGRESSION = {
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

# LightGBM params for classification (taste)
LGB_PARAMS_CLASSIFICATION = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 40,
    "verbose": -1,
}

# Taste class boundaries (binary)
TASTE_THRESHOLD = 4.5
TASTE_CLASSES = {
    "NOT_RECOMMENDED": 0,  # rating < 4.5
    "GOOD": 1,             # rating >= 4.5
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


def rating_to_class(rating: float) -> int:
    """Convert continuous rating to binary labels."""
    return TASTE_CLASSES["GOOD"] if rating >= TASTE_THRESHOLD else TASTE_CLASSES["NOT_RECOMMENDED"]


def class_to_rating(class_label: int) -> float:
    """Convert class label back to representative rating (for compatibility)."""
    return 4.75 if class_label == 1 else 3.5  # GOOD: 4.75, NOT_RECOMMENDED: 3.5


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
    # Convert scores to categories
    true_categories = [nutriscore_to_category(s) for s in y_true]
    pred_categories = [nutriscore_to_category(s) for s in y_pred]
    
    # Exact match accuracy
    exact_accuracy = accuracy_score(true_categories, pred_categories)
    
    # Within-1 category accuracy
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
    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    # Kendall's tau
    kendall_corr, kendall_p = kendalltau(y_true, y_pred)
    
    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'kendall_tau': kendall_corr,
        'kendall_pvalue': kendall_p
    }


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
    print(f"\n  Building TF-IDF features from '{text_column}'...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b'
    )
    
    texts = df[text_column].fillna('')
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"    TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_svd = svd.fit_transform(tfidf_matrix)
    explained_var = svd.explained_variance_ratio_.sum()
    print(f"    SVD reduced to {n_components} components (explained variance: {explained_var:.1%})")
    
    return tfidf_svd.astype(np.float32)


def build_feature_matrices(
    food_df: pd.DataFrame, nutri_df: pd.DataFrame, eco_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
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
    is_classification: bool = False


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)) if np.all(y_true != 0) else 0.0,
        "max_error": float(np.max(np.abs(y_true - y_pred))),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        "explained_variance": float(1 - np.var(y_true - y_pred) / np.var(y_true)),
    }
    return metrics


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    return metrics


def train_classifier(X: np.ndarray, y: np.ndarray, model_name: str) -> ModelArtifacts:
    """Train binary classifier for taste."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    booster = lgb.train(
        LGB_PARAMS_CLASSIFICATION,
        train_set,
        valid_sets=[val_set],
        num_boost_round=4000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # Predict class probabilities for binary classification
    val_pred_proba = booster.predict(X_val)
    val_pred = (val_pred_proba >= 0.5).astype(int)
    
    metrics = calculate_classification_metrics(y_val, val_pred)
    
    model_path = MODEL_DIR / f"{model_name}.txt"
    booster.save_model(model_path)
    
    return ModelArtifacts(
        booster=booster, 
        metrics=metrics, 
        model_path=model_path,
        y_val=y_val,
        val_pred=val_pred,
        is_classification=True
    )


def train_regressor(X: np.ndarray, y: np.ndarray, model_name: str) -> ModelArtifacts:
    """Train regression model for nutrition/sustainability."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    booster = lgb.train(
        LGB_PARAMS_REGRESSION,
        train_set,
        valid_sets=[val_set],
        num_boost_round=4000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    val_pred = booster.predict(X_val)
    metrics = calculate_regression_metrics(y_val, val_pred)
    
    model_path = MODEL_DIR / f"{model_name}.txt"
    booster.save_model(model_path)
    return ModelArtifacts(
        booster=booster, 
        metrics=metrics, 
        model_path=model_path,
        y_val=y_val,
        val_pred=val_pred,
        is_classification=False
    )


def save_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Visualization utilities
# --------------------------------------------------------------------------- #


def plot_classification_evaluation(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    class_names: List[str] = None
) -> None:
    """Create evaluation plots for classification model."""
    if class_names is None:
        class_names = ['NOT_RECOMMENDED', 'GOOD']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{model_name} - Classification Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    
    # 2. Normalized Confusion Matrix
    ax2 = axes[0, 1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Normalized Confusion Matrix')
    
    # 3. Metrics Bar Chart
    ax3 = axes[1, 0]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision_weighted'],
        metrics['recall_weighted'],
        metrics['f1_weighted']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax3.barh(metric_names, metric_values, color=colors, edgecolor='black')
    ax3.set_xlabel('Score')
    ax3.set_title('Evaluation Metrics (Weighted)')
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax3.text(val, i, f' {val:.3f}', va='center', fontweight='bold')
    
    # 4. Classification Report Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics_text = [
        f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "=" * 60,
    ]
    
    for class_name in class_names:
        class_metrics = report[class_name]
        metrics_text.append(
            f"{class_name:<12} "
            f"{class_metrics['precision']:>10.3f} "
            f"{class_metrics['recall']:>10.3f} "
            f"{class_metrics['f1-score']:>10.3f} "
            f"{int(class_metrics['support']):>10,}"
        )
    
    metrics_text.append("=" * 60)
    metrics_text.append(
        f"{'Macro Avg':<12} "
        f"{report['macro avg']['precision']:>10.3f} "
        f"{report['macro avg']['recall']:>10.3f} "
        f"{report['macro avg']['f1-score']:>10.3f} "
        f"{int(report['macro avg']['support']):>10,}"
    )
    metrics_text.append(
        f"{'Weighted Avg':<12} "
        f"{report['weighted avg']['precision']:>10.3f} "
        f"{report['weighted avg']['recall']:>10.3f} "
        f"{report['weighted avg']['f1-score']:>10.3f} "
        f"{int(report['weighted avg']['support']):>10,}"
    )
    
    ax4.text(0.1, 0.9, '\n'.join(metrics_text), 
             fontfamily='monospace', fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Per-Class Metrics', fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"{model_name}_classification_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Saved plot → {plot_path}")
    plt.close()


def plot_regression_evaluation(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    additional_metrics: Dict[str, any] = None
) -> None:
    """Create comprehensive evaluation plots for regression model."""
    
    # Determine layout based on model type and additional metrics
    if additional_metrics:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        gs = None
    
    fig.suptitle(f'{model_name} - Regression Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    if gs:
        ax1 = fig.add_subplot(gs[0, 0])
    else:
        ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals
    if gs:
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
    ax2.set_xlabel('Residuals', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics Bar Chart
    if gs:
        ax3 = fig.add_subplot(gs[0, 2])
    else:
        ax3 = axes[2]
    
    metric_names = ['RMSE', 'MAE', 'R²']
    metric_values = [metrics['rmse'], metrics['mae'], metrics['r2']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax3.barh(metric_names, metric_values, color=colors, edgecolor='black')
    ax3.set_xlabel('Value', fontsize=11)
    ax3.set_title('Key Metrics', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax3.text(val, i, f' {val:.4f}', va='center', fontweight='bold', fontsize=10)
    
    # Additional plots for Nutrition (Nutri-Score categories)
    if model_name == "Nutrition" and additional_metrics and 'nutriscore_metrics' in additional_metrics:
        nutri_metrics = additional_metrics['nutriscore_metrics']
        
        # 4. Confusion Matrix for Nutri-Score categories
        ax4 = fig.add_subplot(gs[1, 0])
        category_order = ['A', 'B', 'C', 'D', 'E']
        cm = confusion_matrix(nutri_metrics['true_categories'], nutri_metrics['pred_categories'], labels=category_order)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=category_order, yticklabels=category_order)
        ax4.set_xlabel('Predicted Category', fontsize=11)
        ax4.set_ylabel('True Category', fontsize=11)
        ax4.set_title('Nutri-Score Category Confusion Matrix', fontsize=12, fontweight='bold')
        
        # 5. Category accuracy metrics
        ax5 = fig.add_subplot(gs[1, 1])
        accuracy_metrics = ['Exact Match', 'Within 1 Category']
        accuracy_values = [nutri_metrics['exact_accuracy'], nutri_metrics['within_1_accuracy']]
        bars5 = ax5.bar(accuracy_metrics, accuracy_values, color=['#27ae60', '#f39c12'], edgecolor='black')
        ax5.set_ylabel('Accuracy', fontsize=11)
        ax5.set_title('Nutri-Score Category Accuracy', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars5, accuracy_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 6. Metrics summary table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        metrics_text = [
            f"{'Metric':<20} {'Value':>12}",
            "=" * 34,
            f"{'RMSE':<20} {metrics['rmse']:>12.4f}",
            f"{'MAE':<20} {metrics['mae']:>12.4f}",
            f"{'R² Score':<20} {metrics['r2']:>12.4f}",
            "=" * 34,
            f"{'Category Exact':<20} {nutri_metrics['exact_accuracy']:>12.1%}",
            f"{'Within-1 Category':<20} {nutri_metrics['within_1_accuracy']:>12.1%}",
            "=" * 34,
            f"{'Samples':<20} {len(y_true):>12,}",
        ]
        ax6.text(0.1, 0.5, '\n'.join(metrics_text),
                fontfamily='monospace', fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_title('Summary Statistics', fontweight='bold')
    
    # Additional plots for Sustainability (Eco-Score ranking)
    elif model_name == "Sustainability" and additional_metrics and 'ecoscore_metrics' in additional_metrics:
        eco_metrics = additional_metrics['ecoscore_metrics']
        
        # 4. Rank correlation metrics
        ax4 = fig.add_subplot(gs[1, 0])
        rank_metrics = ['Spearman ρ', 'Kendall τ']
        rank_values = [eco_metrics['spearman_correlation'], eco_metrics['kendall_tau']]
        bars4 = ax4.bar(rank_metrics, rank_values, color=['#8e44ad', '#e74c3c'], edgecolor='black')
        ax4.set_ylabel('Correlation', fontsize=11)
        ax4.set_title('Eco-Score Ranking Correlation', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars4, rank_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 5. Scatter with error heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        errors = np.abs(y_true - y_pred)
        scatter = ax5.scatter(y_true, y_pred, c=errors, cmap='RdYlGn_r', alpha=0.6, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax5.set_xlabel('True Eco-Score', fontsize=11)
        ax5.set_ylabel('Predicted Eco-Score', fontsize=11)
        ax5.set_title('Eco-Score: Error Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax5, label='Absolute Error')
        ax5.grid(True, alpha=0.3)
        
        # 6. Metrics summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        metrics_text = [
            f"{'Metric':<20} {'Value':>12}",
            "=" * 34,
            f"{'RMSE':<20} {metrics['rmse']:>12.4f}",
            f"{'MAE':<20} {metrics['mae']:>12.4f}",
            f"{'R² Score':<20} {metrics['r2']:>12.4f}",
            "=" * 34,
            f"{'Spearman ρ':<20} {eco_metrics['spearman_correlation']:>12.4f}",
            f"{'Kendall τ':<20} {eco_metrics['kendall_tau']:>12.4f}",
            "=" * 34,
            f"{'Samples':<20} {len(y_true):>12,}",
        ]
        ax6.text(0.1, 0.5, '\n'.join(metrics_text),
                fontfamily='monospace', fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_title('Summary Statistics', fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / f"{model_name}_evaluation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Saved plot → {plot_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# Recommendation utilities
# --------------------------------------------------------------------------- #


def calculate_pantry_coverage(ingredient_text: str, user_pantry: Set[str]) -> float:
    if not ingredient_text or pd.isna(ingredient_text):
        return 0.0
    
    recipe_ingredients = [ing.strip().lower() for ing in str(ingredient_text).split(',')]
    
    if not recipe_ingredients:
        return 0.0
    
    matched = sum(
        1 for recipe_ing in recipe_ingredients
        if any(pantry_item.lower() in recipe_ing for pantry_item in user_pantry)
    )
    
    return matched / len(recipe_ingredients)


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scores to 0-1 scale."""
    df = df.copy()
    
    # Taste: already 0-5 scale from class probabilities
    df['norm_taste'] = df['pred_taste'] / 5.0
    
    # Nutrition: invert so lower is better
    df['norm_nutrition'] = 1.0 - ((df['pred_nutriscore'] + 15.0) / 55.0)
    
    # Sustainability
    df['norm_sustainability'] = df['pred_ecoscore'] / 100.0
    
    df['norm_taste'] = df['norm_taste'].clip(0, 1)
    df['norm_nutrition'] = df['norm_nutrition'].clip(0, 1)
    df['norm_sustainability'] = df['norm_sustainability'].clip(0, 1)
    
    return df


def calculate_precision_at_k(recommended_df: pd.DataFrame, k: int, good_threshold: float = 4.5) -> float:
    """
    Calculate Precision@K: proportion of recommended items that are actually GOOD.
    
    Args:
        recommended_df: DataFrame with recommendations (must have 'avg_rating' column)
        k: Number of top items to consider
        good_threshold: Minimum rating to be considered GOOD (default: 4.5)
    
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if len(recommended_df) == 0:
        return 0.0
    
    top_k = recommended_df.head(k)
    n_good = (top_k['avg_rating'] >= good_threshold).sum()
    return n_good / min(k, len(top_k))


def calculate_ndcg_at_k(recommended_df: pd.DataFrame, k: int, good_threshold: float = 4.5) -> float:
    """
    Calculate nDCG@K: ranking quality metric considering position.
    Higher ratings at top positions = higher nDCG.
    
    Args:
        recommended_df: DataFrame with recommendations (must have 'avg_rating' column)
        k: Number of top items to consider
        good_threshold: Minimum rating to be considered GOOD (default: 4.5)
    
    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    if len(recommended_df) == 0:
        return 0.0
    
    top_k = recommended_df.head(k)
    
    # True relevance: actual ratings
    y_true = top_k['avg_rating'].values.reshape(1, -1)
    
    # Predicted relevance: use final_score (our ranking score)
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
    
    Args:
        recipes_df: Full recipe dataframe
        user_pantry: User's pantry items
        min_coverage: Minimum coverage threshold
        k: Number of recommendations
        n_samples: Number of random samples to average (default: 100)
    
    Returns:
        Dictionary with baseline metrics
    """
    # Filter by coverage
    df = recipes_df.copy()
    df['pantry_coverage'] = df['ingredient_text'].apply(
        lambda x: calculate_pantry_coverage(x, user_pantry)
    )
    eligible = df[df['pantry_coverage'] >= min_coverage].copy()
    
    if len(eligible) < k:
        return {'nutriscore': 0, 'ecoscore': 0, 'avg_rating': 0}
    
    # Sample multiple times and average
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
    """
    Create comprehensive visualization of recommendation quality metrics.
    
    Generates 4 subplots:
      1. Precision@K comparison
      2. nDCG@K comparison
      3. Multi-objective scores (Nutrition, Eco, Taste) vs Baseline
      4. Score improvements bar chart
    """
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
    df = recipes_df.copy()
    
    print(f"\n=== Filtering by pantry coverage (>= {min_coverage:.0%}) ===")
    df['pantry_coverage'] = df['ingredient_text'].apply(
        lambda x: calculate_pantry_coverage(x, user_pantry)
    )
    
    df = df[df['pantry_coverage'] >= min_coverage].copy()
    print(f"Recipes with >= {min_coverage:.0%} coverage: {len(df):,}")
    
    if len(df) == 0:
        print("⚠️  No recipes meet the coverage threshold!")
        return pd.DataFrame()
    
    df = normalize_scores(df)
    
    df['final_score'] = (
        weight_taste * df['norm_taste'] +
        weight_nutrition * df['norm_nutrition'] +
        weight_sustainability * df['norm_sustainability']
    )
    
    df = df.sort_values('final_score', ascending=False).head(top_k)
    
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

    print("\n=== Converting taste ratings to binary classes ===")
    print(f"Threshold: {TASTE_THRESHOLD} stars")
    food_df['taste_class'] = food_df['avg_rating'].apply(rating_to_class)
    class_dist = food_df['taste_class'].value_counts().sort_index()
    print(f"Class distribution:")
    print(f"  NOT_RECOMMENDED (0): {class_dist[0]:6,} ({class_dist[0]/len(food_df)*100:5.1f}%)")
    print(f"  GOOD (1):            {class_dist[1]:6,} ({class_dist[1]/len(food_df)*100:5.1f}%)")

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

    # Taste features
    extra_feats = food_df[TASTE_EXTRA_FEATURES].fillna(0).values.astype(np.float32)
    taste_X = np.hstack([food_macro, extra_feats, tfidf_food])
    print(f"  Taste features shape: {taste_X.shape}")
    
    nutri_X_enhanced = np.hstack([nutri_X, tfidf_nutri])
    eco_X_enhanced = np.hstack([eco_X, tfidf_eco])
    print(f"  Nutrition features shape: {nutri_X_enhanced.shape}")
    print(f"  Sustainability features shape: {eco_X_enhanced.shape}")

    print("\n=== Training models ===")
    print("\n[1/3] Training Taste Binary Classifier...")
    taste_artifacts = train_classifier(taste_X, food_df["taste_class"].values, "taste_binary_classifier")
    print(f"  ✓ Taste metrics: Accuracy={taste_artifacts.metrics['accuracy']:.4f}, F1={taste_artifacts.metrics['f1_weighted']:.4f}, Precision={taste_artifacts.metrics['precision_weighted']:.4f}, Recall={taste_artifacts.metrics['recall_weighted']:.4f}")

    print("\n[2/3] Training Nutrition Regressor...")
    y_nutri = nutri_df["nutriscore_score"].clip(-15, 40).values
    nutrition_artifacts = train_regressor(nutri_X_enhanced, y_nutri, "nutrition_model")
    print(f"  ✓ Nutrition metrics: RMSE={nutrition_artifacts.metrics['rmse']:.4f}, R²={nutrition_artifacts.metrics['r2']:.4f}")

    print("\n[3/3] Training Sustainability Regressor...")
    y_eco = eco_df["ecoscore_score"].clip(0, 100).values
    sustainability_artifacts = train_regressor(eco_X_enhanced, y_eco, "sustainability_model")
    print(f"  ✓ Sustainability metrics: RMSE={sustainability_artifacts.metrics['rmse']:.4f}, R²={sustainability_artifacts.metrics['r2']:.4f}")

    # Save all metrics
    all_metrics = {
        "Taste_Classification": taste_artifacts.metrics,
        "Nutrition": nutrition_artifacts.metrics,
        "Sustainability": sustainability_artifacts.metrics,
    }
    save_json(all_metrics, OUTPUT_DIR / "training_metrics_classification.json")
    
    # Generate evaluation plots
    print("\n=== Generating evaluation plots ===")
    plot_classification_evaluation("Taste_Binary", taste_artifacts.y_val, taste_artifacts.val_pred, 
                                   taste_artifacts.metrics, class_names=['NOT_RECOMMENDED', 'GOOD'])
    
    # Calculate Nutri-Score category metrics
    nutriscore_metrics = calculate_nutriscore_metrics(nutrition_artifacts.y_val, nutrition_artifacts.val_pred)
    print(f"  ℹ️  Nutri-Score Category Accuracy: {nutriscore_metrics['exact_accuracy']:.1%}, Within-1: {nutriscore_metrics['within_1_accuracy']:.1%}")
    
    plot_regression_evaluation("Nutrition", nutrition_artifacts.y_val, nutrition_artifacts.val_pred,
                              nutrition_artifacts.metrics, 
                              additional_metrics={'nutriscore_metrics': nutriscore_metrics})
    
    # Calculate Eco-Score ranking metrics
    ecoscore_metrics = calculate_ecoscore_rank_metrics(sustainability_artifacts.y_val, sustainability_artifacts.val_pred)
    print(f"  ℹ️  Eco-Score Spearman ρ: {ecoscore_metrics['spearman_correlation']:.4f}, Kendall τ: {ecoscore_metrics['kendall_tau']:.4f}")
    
    plot_regression_evaluation("Sustainability", sustainability_artifacts.y_val, sustainability_artifacts.val_pred,
                              sustainability_artifacts.metrics,
                              additional_metrics={'ecoscore_metrics': ecoscore_metrics})

    print("\n=== Generating Food.com predictions ===")
    # Predict taste class probabilities (binary)
    taste_proba = taste_artifacts.booster.predict(taste_X)
    taste_class = (taste_proba >= 0.5).astype(int)
    
    # Convert class back to rating scale (0-5) for compatibility
    food_df["pred_taste_class"] = taste_class
    food_df["pred_taste"] = [class_to_rating(c) for c in taste_class]
    food_df["pred_taste_proba"] = taste_proba  # Save probability for analysis
    
    # Predict nutrition and sustainability
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
        "id", "name", "avg_rating", "taste_class", "pred_taste_class", "pred_taste", "pred_taste_proba",
        "pred_nutriscore", "pred_ecoscore", "minutes", "n_steps", "n_ingredients",
        "ingredient_text", "calories", "fat", "sat_fat", "carbs", "sugar", "protein", "sodium",
    ]
    food_df_filtered[output_cols].to_csv(
        OUTPUT_DIR / "recipes_with_predictions_classification.csv", index=False
    )
    print(f"Saved predictions → {OUTPUT_DIR / 'recipes_with_predictions_classification.csv'}")
    
    # Use filtered dataframe for recommendations
    food_df = food_df_filtered
    
    # Recommendation demo
    print("\n" + "="*80)
    print("RECIPE RECOMMENDATION DEMO (Binary Classification)")
    print("="*80)
    
    user_pantry = {
        'beef', 'pasta', 'tomato', 'olive oil', 'garlic',
        'basil', 'egg', 'flour', 'sugar', 'salt', 'pepper'
        'spinach', 'red pepper', 'chicken stock'
    }
    
    print(f"\n📦 User's Pantry ({len(user_pantry)} items):")
    print(f"   {', '.join(sorted(user_pantry))}")
    
    recommendations = recommend_recipes(
        recipes_df=food_df,
        user_pantry=user_pantry,
        weight_taste=0.4,
        weight_nutrition=0.3,
        weight_sustainability=0.3,
        min_coverage=0.8,
        top_k=10
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
        
        class_labels = {0: 'NOT_REC', 1: 'GOOD'}
        
        for rank, row in enumerate(recommendations.itertuples(), 1):
            print(f"\n🏆 Rank #{rank}: {row.name}")
            print(f"   Final Score: {row.final_score:.3f}")
            print(f"   ├─ 😋 Taste:          {class_labels[row.pred_taste_class]} (prob={row.pred_taste_proba:.3f}, actual={row.avg_rating:.2f}⭐)")
            print(f"   ├─ 🥗 Nutrition:      {row.pred_nutriscore:.1f} (Nutri-Score)")
            print(f"   ├─ 🌱 Sustainability: {row.pred_ecoscore:.1f}/100 (Eco-Score)")
            print(f"   ├─ 📦 Pantry Coverage: {row.pantry_coverage:.1%}")
            print(f"   └─ 🛒 Missing: {len(row.missing_ingredients)} items")
        
        output_cols_rec = [
            "id", "name", "final_score", "taste_class", "pred_taste_class", "pred_taste", "pred_taste_proba",
            "pred_nutriscore", "pred_ecoscore", "norm_taste", "norm_nutrition", 
            "norm_sustainability", "pantry_coverage", "minutes", "n_steps", 
            "n_ingredients", "missing_ingredients", "ingredient_text"
        ]
        recommendations[output_cols_rec].to_csv(
            OUTPUT_DIR / "top_recommendations_classification.csv", index=False
        )
        print(f"\n💾 Saved recommendations → {OUTPUT_DIR / 'top_recommendations_classification.csv'}")
    
    print("\n" + "="*80)
    print("✅ Pipeline complete (Binary Classification)!")
    print("="*80)


if __name__ == "__main__":
    main()
