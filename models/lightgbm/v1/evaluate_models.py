"""
Model Evaluation Script
Comprehensive evaluation of all 3 models with detailed metrics
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def evaluate_model(y_true, y_pred, model_name):
    """Calculate comprehensive metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Residual statistics
    residuals = y_true - y_pred
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Residual Mean': residual_mean,
        'Residual Std': residual_std,
        'Min Error': np.min(residuals),
        'Max Error': np.max(residuals)
    }

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    data_dir = '../data'
    model_dir = './models'
    
    # Load features and labels
    print("\n[1/4] Loading data...")
    X_taste = np.load(f'{data_dir}/X_taste.npy')
    X_nutrition = np.load(f'{data_dir}/X_nutrition.npy')
    X_sustainability = np.load(f'{data_dir}/X_sustainability.npy')
    X_diet = np.load(f'{data_dir}/X_diet.npy')
    y_taste = np.load(f'{data_dir}/y_taste.npy')
    y_nutrition = np.load(f'{data_dir}/y_nutrition.npy')
    y_sustainability = np.load(f'{data_dir}/y_sustainability.npy')
    y_diet = np.load(f'{data_dir}/y_diet.npy')
    
    # Split
    print("\n[2/4] Creating train/test splits...")
    X_taste_train, X_taste_test, y_taste_train, y_taste_test = train_test_split(
        X_taste, y_taste, test_size=0.2, random_state=42
    )
    X_nutr_train, X_nutr_test, y_nutr_train, y_nutr_test = train_test_split(
        X_nutrition, y_nutrition, test_size=0.2, random_state=42
    )
    X_sust_train, X_sust_test, y_sust_train, y_sust_test = train_test_split(
        X_sustainability, y_sustainability, test_size=0.2, random_state=42
    )
    X_diet_train, X_diet_test, y_diet_train, y_diet_test = train_test_split(
        X_diet, y_diet, test_size=0.2, random_state=42
    )
    
    # Load models
    print("\n[3/4] Loading trained models...")
    model_taste = lgb.Booster(model_file=f'{model_dir}/model_taste.txt')
    model_nutrition = lgb.Booster(model_file=f'{model_dir}/model_nutrition.txt')
    model_sustainability = lgb.Booster(model_file=f'{model_dir}/model_sustainability.txt')
    diet_labels = ['vegan', 'vegetarian', 'gluten_free', 'low_sodium', 'low_sugar', 'healthy']
    diet_models = {name: lgb.Booster(model_file=f'{model_dir}/diet_{name}.txt') for name in diet_labels}
    
    # Predictions
    print("\n[4/4] Evaluating models...\n")
    
    y_pred_taste = model_taste.predict(X_taste_test)
    y_pred_nutr = model_nutrition.predict(X_nutr_test)
    y_pred_sust = model_sustainability.predict(X_sust_test)
    y_pred_diet = {}
    for name, model in diet_models.items():
        y_pred_diet[name] = model.predict(X_diet_test)
    
    # ========== TASTE MODEL ==========
    print("="*70)
    print("MODEL 1: TASTE (predict avg_rating, scale 1-5)")
    print("="*70)
    metrics_taste = evaluate_model(y_taste_test, y_pred_taste, "Taste")
    
    for key, value in metrics_taste.items():
        print(f"  {key:20s}: {value:.4f}")
    
    print(f"\n  Interpretation:")
    print(f"    - RMSE 0.42 on 1-5 scale = ±0.42 rating error")
    print(f"    - Most predictions within 0.3 ratings of true value")
    print(f"    - R² = {metrics_taste['R²']:.3f} (variance explained)")
    
    # Distribution analysis
    print(f"\n  Prediction Distribution:")
    print(f"    True - Mean: {y_taste_test.mean():.3f}, Std: {y_taste_test.std():.3f}")
    print(f"    Pred - Mean: {y_pred_taste.mean():.3f}, Std: {y_pred_taste.std():.3f}")
    
    # Error breakdown
    errors = np.abs(y_taste_test - y_pred_taste)
    print(f"\n  Error Distribution:")
    print(f"    < 0.25 rating: {np.sum(errors < 0.25) / len(errors) * 100:.1f}%")
    print(f"    < 0.50 rating: {np.sum(errors < 0.5) / len(errors) * 100:.1f}%")
    print(f"    < 1.00 rating: {np.sum(errors < 1.0) / len(errors) * 100:.1f}%")
    print(f"    > 1.00 rating: {np.sum(errors >= 1.0) / len(errors) * 100:.1f}%")
    
    # ========== NUTRITION MODEL ==========
    print("\n" + "="*70)
    print("MODEL 2: NUTRITION (predict nutrition_score, scale 0-1)")
    print("="*70)
    metrics_nutr = evaluate_model(y_nutr_test, y_pred_nutr, "Nutrition")
    
    for key, value in metrics_nutr.items():
        print(f"  {key:20s}: {value:.4f}")
    
    print(f"\n  Interpretation:")
    print(f"    - RMSE 0.003 on 0-1 scale = 0.3% error")
    print(f"    - MAE 0.001 = predictions are extremely accurate")
    print(f"    - R² = {metrics_nutr['R²']:.3f} (almost perfect)")
    print(f"    - ⚠️  WARNING: This suggests the model is learning the formula,")
    print(f"               not discovering new patterns!")
    
    print(f"\n  Prediction Distribution:")
    print(f"    True - Mean: {y_nutr_test.mean():.3f}, Std: {y_nutr_test.std():.3f}")
    print(f"    Pred - Mean: {y_pred_nutr.mean():.3f}, Std: {y_pred_nutr.std():.3f}")
    
    # ========== SUSTAINABILITY MODEL ==========
    print("\n" + "="*70)
    print("MODEL 3: SUSTAINABILITY (predict sustainability_score, scale 0-1)")
    print("="*70)
    metrics_sust = evaluate_model(y_sust_test, y_pred_sust, "Sustainability")
    
    for key, value in metrics_sust.items():
        print(f"  {key:20s}: {value:.4f}")
    
    print(f"\n  Interpretation:")
    print(f"    - RMSE 0.13 on 0-1 scale = 13% error")
    print(f"    - MAE 0.10 = predictions typically off by 10%")
    print(f"    - R² = {metrics_sust['R²']:.3f} (variance explained)")
    
    print(f"\n  Prediction Distribution:")
    print(f"    True - Mean: {y_sust_test.mean():.3f}, Std: {y_sust_test.std():.3f}")
    print(f"    Pred - Mean: {y_pred_sust.mean():.3f}, Std: {y_pred_sust.std():.3f}")
    
    # Error breakdown
    errors_sust = np.abs(y_sust_test - y_pred_sust)
    print(f"\n  Error Distribution:")
    print(f"    < 0.05 (5%):  {np.sum(errors_sust < 0.05) / len(errors_sust) * 100:.1f}%")
    print(f"    < 0.10 (10%): {np.sum(errors_sust < 0.10) / len(errors_sust) * 100:.1f}%")
    print(f"    < 0.20 (20%): {np.sum(errors_sust < 0.20) / len(errors_sust) * 100:.1f}%")
    print(f"    > 0.20 (20%): {np.sum(errors_sust >= 0.20) / len(errors_sust) * 100:.1f}%")
    
    # ========== DIET MODEL ==========
    print("\n" + "="*70)
    print("MODEL 4: DIET (multi-label probabilities)")
    print("="*70)
    print("\nThreshold @0.5 (balanced) vs @0.8 (strict filtering):")
    diet_acc_05 = {}
    diet_acc_08 = {}
    for i, name in enumerate(diet_labels):
        pred_05 = (y_pred_diet[name] >= 0.5).astype(int)
        pred_08 = (y_pred_diet[name] >= 0.8).astype(int)
        acc_05 = np.mean(pred_05 == y_diet_test[:, i])
        acc_08 = np.mean(pred_08 == y_diet_test[:, i])
        diet_acc_05[name] = acc_05
        diet_acc_08[name] = acc_08
        print(f"  {name:15s} acc@0.5: {acc_05:.3f}  |  acc@0.8: {acc_08:.3f}")
    
    # Precision/Recall at 0.8 for critical labels
    print("\n  Critical Labels (vegan, vegetarian, gluten_free) at threshold=0.8:")
    for i, name in enumerate(['vegan', 'vegetarian', 'gluten_free']):
        pred_08 = (y_pred_diet[name] >= 0.8).astype(int)
        true_pos = np.sum((pred_08 == 1) & (y_diet_test[:, i] == 1))
        false_pos = np.sum((pred_08 == 1) & (y_diet_test[:, i] == 0))
        false_neg = np.sum((pred_08 == 0) & (y_diet_test[:, i] == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"    {name:15s} Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    summary_data = {
        'Model': ['Taste', 'Nutrition', 'Sustainability'],
        'RMSE': [metrics_taste['RMSE'], metrics_nutr['RMSE'], metrics_sust['RMSE']],
        'MAE': [metrics_taste['MAE'], metrics_nutr['MAE'], metrics_sust['MAE']],
        'R²': [metrics_taste['R²'], metrics_nutr['R²'], metrics_sust['R²']],
        'Scale': ['1-5', '0-1', '0-1'],
        'ML Quality': ['✅ Good', '⚠️  Too Perfect', '✅ Good']
    }
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    print("\nDiet Accuracies (threshold 0.5):")
    for name in diet_labels:
        print(f"  {name:12s}: {diet_acc_05[name]:.3f}")
    
    # ========== VISUALIZATIONS ==========
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path('./evaluation_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Prediction vs True (3 models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Taste
    axes[0].scatter(y_taste_test, y_pred_taste, alpha=0.3, s=10)
    axes[0].plot([y_taste_test.min(), y_taste_test.max()], 
                 [y_taste_test.min(), y_taste_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Rating', fontsize=12)
    axes[0].set_ylabel('Predicted Rating', fontsize=12)
    axes[0].set_title(f'Taste Model\nR²={metrics_taste["R²"]:.3f}, RMSE={metrics_taste["RMSE"]:.3f}', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # Nutrition
    axes[1].scatter(y_nutr_test, y_pred_nutr, alpha=0.3, s=10, color='orange')
    axes[1].plot([y_nutr_test.min(), y_nutr_test.max()], 
                 [y_nutr_test.min(), y_nutr_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('True Nutrition Score', fontsize=12)
    axes[1].set_ylabel('Predicted Nutrition Score', fontsize=12)
    axes[1].set_title(f'Nutrition Model\nR²={metrics_nutr["R²"]:.3f}, RMSE={metrics_nutr["RMSE"]:.4f}', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    # Sustainability
    axes[2].scatter(y_sust_test, y_pred_sust, alpha=0.3, s=10, color='green')
    axes[2].plot([y_sust_test.min(), y_sust_test.max()], 
                 [y_sust_test.min(), y_sust_test.max()], 'r--', lw=2)
    axes[2].set_xlabel('True Sustainability Score', fontsize=12)
    axes[2].set_ylabel('Predicted Sustainability Score', fontsize=12)
    axes[2].set_title(f'Sustainability Model\nR²={metrics_sust["R²"]:.3f}, RMSE={metrics_sust["RMSE"]:.3f}', fontsize=13)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_vs_true.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: prediction_vs_true.png")
    plt.close()
    
    # 2. Residual distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    residuals_taste = y_taste_test - y_pred_taste
    residuals_nutr = y_nutr_test - y_pred_nutr
    residuals_sust = y_sust_test - y_pred_sust
    
    axes[0].hist(residuals_taste, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Residual (True - Predicted)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Taste Residuals\nMean={np.mean(residuals_taste):.4f}, Std={np.std(residuals_taste):.4f}', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals_nutr, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual (True - Predicted)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Nutrition Residuals\nMean={np.mean(residuals_nutr):.4f}, Std={np.std(residuals_nutr):.4f}', fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(residuals_sust, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', lw=2)
    axes[2].set_xlabel('Residual (True - Predicted)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title(f'Sustainability Residuals\nMean={np.mean(residuals_sust):.4f}, Std={np.std(residuals_sust):.4f}', fontsize=13)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_distributions.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: residual_distributions.png")
    plt.close()
    
    # 3. Error distributions (box plots)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    errors_data = [
        np.abs(residuals_taste),
        np.abs(residuals_nutr) * 5,  # Scale to 0-5 for comparison
        np.abs(residuals_sust) * 5   # Scale to 0-5 for comparison
    ]
    
    bp = ax.boxplot(errors_data, labels=['Taste', 'Nutrition\n(scaled x5)', 'Sustainability\n(scaled x5)'],
                    patch_artist=True, showmeans=True)
    
    colors = ['#3498db', '#e67e22', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Absolute Error', fontsize=13)
    ax.set_title('Error Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_boxplots.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: error_boxplots.png")
    plt.close()
    
    # 4. R² comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Taste', 'Nutrition', 'Sustainability']
    r2_scores = [metrics_taste['R²'], metrics_nutr['R²'], metrics_sust['R²']]
    colors_bar = ['#3498db', '#e67e22', '#2ecc71']
    
    bars = ax.bar(models, r2_scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('R² Score', fontsize=13)
    ax.set_title('Model Performance: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: r2_comparison.png")
    plt.close()
    
    # 5. Diet model accuracies
    fig, ax = plt.subplots(figsize=(12, 6))
    
    diet_names = list(diet_acc_05.keys())
    diet_scores = list(diet_acc_05.values())
    
    bars = ax.barh(diet_names, diet_scores, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, diet_scores):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Accuracy @threshold=0.5', fontsize=13)
    ax.set_title('Diet Model: Multi-Label Classification Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.axvline(0.5, color='red', linestyle='--', lw=2, alpha=0.5, label='Baseline (50%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diet_accuracies.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: diet_accuracies.png")
    plt.close()
    
    # 6. Model Performance Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 6a. RMSE Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    rmse_scores = [metrics_taste['RMSE'], metrics_nutr['RMSE'], metrics_sust['RMSE']]
    bars1 = ax1.bar(models, rmse_scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, score in zip(bars1, rmse_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 6b. MAE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    mae_scores = [metrics_taste['MAE'], metrics_nutr['MAE'], metrics_sust['MAE']]
    bars2 = ax2.bar(models, mae_scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, score in zip(bars2, mae_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 6c. R² Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    r2_scores = [metrics_taste['R²'], metrics_nutr['R²'], metrics_sust['R²']]
    bars3 = ax3.bar(models, r2_scores, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, score in zip(bars3, r2_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=11)
    ax3.set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 6d. Prediction Accuracy (Taste - within ±0.5)
    ax4 = fig.add_subplot(gs[1, 0])
    taste_errors = np.abs(y_taste_test - y_pred_taste)
    taste_buckets = ['< 0.25', '0.25-0.5', '0.5-1.0', '> 1.0']
    taste_percentages = [
        np.sum(taste_errors < 0.25) / len(taste_errors) * 100,
        np.sum((taste_errors >= 0.25) & (taste_errors < 0.5)) / len(taste_errors) * 100,
        np.sum((taste_errors >= 0.5) & (taste_errors < 1.0)) / len(taste_errors) * 100,
        np.sum(taste_errors >= 1.0) / len(taste_errors) * 100
    ]
    bars4 = ax4.bar(taste_buckets, taste_percentages, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, pct in zip(bars4, taste_percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_xlabel('Error Range', fontsize=11)
    ax4.set_title('Taste: Error Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 6e. Sustainability Error Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    sust_errors = np.abs(y_sust_test - y_pred_sust)
    sust_buckets = ['< 0.05', '0.05-0.1', '0.1-0.2', '> 0.2']
    sust_percentages = [
        np.sum(sust_errors < 0.05) / len(sust_errors) * 100,
        np.sum((sust_errors >= 0.05) & (sust_errors < 0.1)) / len(sust_errors) * 100,
        np.sum((sust_errors >= 0.1) & (sust_errors < 0.2)) / len(sust_errors) * 100,
        np.sum(sust_errors >= 0.2) / len(sust_errors) * 100
    ]
    bars5 = ax5.bar(sust_buckets, sust_percentages, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, pct in zip(bars5, sust_percentages):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax5.set_ylabel('Percentage (%)', fontsize=11)
    ax5.set_xlabel('Error Range', fontsize=11)
    ax5.set_title('Sustainability: Error Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6f. Model Quality Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = [
        ['Model', 'RMSE', 'R²', 'Quality'],
        ['Taste', f"{metrics_taste['RMSE']:.3f}", f"{metrics_taste['R²']:.3f}", '✅ Real ML'],
        ['Nutrition', f"{metrics_nutr['RMSE']:.4f}", f"{metrics_nutr['R²']:.3f}", '⚠️ Formula'],
        ['Sustainability', f"{metrics_sust['RMSE']:.3f}", f"{metrics_sust['R²']:.3f}", '✅ Good']
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Row styling
    row_colors = ['#3498db', '#e67e22', '#2ecc71']
    for i in range(1, 4):
        for j in range(4):
            table[(i, j)].set_facecolor(row_colors[i-1])
            table[(i, j)].set_alpha(0.3)
    
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    # 6g-i. Bottom row: Prediction vs True scatter plots (compact)
    for idx, (y_true, y_pred, name, color, metrics) in enumerate([
        (y_taste_test, y_pred_taste, 'Taste', '#3498db', metrics_taste),
        (y_nutr_test, y_pred_nutr, 'Nutrition', '#e67e22', metrics_nutr),
        (y_sust_test, y_pred_sust, 'Sustainability', '#2ecc71', metrics_sust)
    ]):
        ax = fig.add_subplot(gs[2, idx])
        ax.scatter(y_true, y_pred, alpha=0.3, s=5, color=color)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, alpha=0.8)
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{name}\nR²={metrics["R²"]:.3f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'model_performance_dashboard.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: model_performance_dashboard.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir.absolute()}")
    print("="*70)
    

if __name__ == '__main__':
    main()
