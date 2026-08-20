import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def evaluate_recommendation_metrics_with_viz(embeddings, csv_path, top_k_values=[1, 5, 10, 20], num_queries=100):
    print("\n" + "="*70)
    print("RECOMMENDATION METRICS EVALUATION WITH VISUALIZATIONS")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    all_indices = np.arange(len(embeddings))
    if len(embeddings) > num_queries:
        query_indices = np.random.choice(all_indices, size=num_queries, replace=False)
    else:
        query_indices = all_indices
    
    print(f"\nTesting on {len(query_indices)} random recipes...")
    
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    all_hr = {k: [] for k in top_k_values}
    all_mrr = {k: [] for k in top_k_values}
    all_precision = {k: [] for k in top_k_values}
    
    results = {}
    
    for k in top_k_values:
        print(f"Computing metrics @k={k}...")
        
        hr_k_list = []
        mrr_k_list = []
        p_k_list = []
        
        for query_idx in query_indices:
            sims = similarity_matrix[query_idx].copy()
            sims[query_idx] = -np.inf
            top_k_indices = np.argsort(-sims)[:k]
            
            target_ing_count = len(str(df.iloc[query_idx]["Normalized_Ingredients"]).split(','))
            
            relevant_found = False
            reciprocal_rank = 0.0
            relevant_count = 0
            
            for rank, rec_idx in enumerate(top_k_indices, 1):
                rec_ing_count = len(str(df.iloc[rec_idx]["Normalized_Ingredients"]).split(','))
                if abs(rec_ing_count - target_ing_count) / target_ing_count <= 0.2:
                    if not relevant_found:
                        reciprocal_rank = 1.0 / rank
                        relevant_found = True
                    relevant_count += 1
            
            hr_k_list.append(1 if relevant_found else 0)
            mrr_k_list.append(reciprocal_rank)
            p_k_list.append(relevant_count / k)
        
        all_hr[k] = hr_k_list
        all_mrr[k] = mrr_k_list
        all_precision[k] = p_k_list
        
        results[f'HR@{k}'] = np.mean(hr_k_list)
        results[f'MRR@{k}'] = np.mean(mrr_k_list)
        results[f'P@{k}'] = np.mean(p_k_list)
        
        print(f"  HR@{k}: {np.mean(hr_k_list)*100:.2f}% | MRR@{k}: {np.mean(mrr_k_list):.4f} | P@{k}: {np.mean(p_k_list):.4f}")
    
    fig = create_metrics_visualizations(results, all_hr, all_mrr, all_precision, top_k_values)
    
    print("="*70)
    return results, fig


def create_metrics_visualizations(results, all_hr, all_mrr, all_precision, top_k_values):
    figures = {}
    k_values = top_k_values
    hr_values = [results[f'HR@{k}'] for k in k_values]
    mrr_values = [results[f'MRR@{k}'] for k in k_values]
    p_values = [results[f'P@{k}'] for k in k_values]
    
    # ========== FIGURE 1: Hit Rate Trend ==========
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, [v*100 for v in hr_values], 'o-', linewidth=3, markersize=10, color='#2E86AB', label='Hit Rate')
    ax.fill_between(k_values, 0, [v*100 for v in hr_values], alpha=0.2, color='#2E86AB')
    ax.set_xlabel('K (Number of Recommendations)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hit Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Hit Rate @ K\n(% of queries with ≥1 relevant item)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    for k, v in zip(k_values, hr_values):
        ax.text(k, v*100 + 2, f'{v*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    figures['hit_rate'] = fig1
    
    # ========== FIGURE 2: MRR Trend ==========
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, mrr_values, 's-', linewidth=3, markersize=10, color='#A23B72', label='MRR')
    ax.fill_between(k_values, 0, mrr_values, alpha=0.2, color='#A23B72')
    ax.set_xlabel('K (Number of Recommendations)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Reciprocal Rank', fontsize=13, fontweight='bold')
    ax.set_title('Mean Reciprocal Rank @ K\n(how early is the first relevant item?)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(mrr_values) + 0.1)
    for k, v in zip(k_values, mrr_values):
        ax.text(k, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    figures['mrr'] = fig2
    
    # ========== FIGURE 3: Precision Trend ==========
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, p_values, '^-', linewidth=3, markersize=10, color='#F18F01', label='Precision')
    ax.fill_between(k_values, 0, p_values, alpha=0.2, color='#F18F01')
    ax.set_xlabel('K (Number of Recommendations)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision @ K\n(% of recommendations that are relevant)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(p_values) + 0.05)
    for k, v in zip(k_values, p_values):
        ax.text(k, v + 0.01, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    figures['precision'] = fig3
    
    # ========== FIGURE 4: HR Distribution ==========
    fig4, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(k_values))
    data_hr = [all_hr[k] for k in k_values]
    bp1 = ax.boxplot(data_hr, positions=positions, patch_artist=True, widths=0.6,
                      boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=11, fontweight='bold')
    ax.set_ylabel('Hit Rate (0 or 1)', fontsize=13, fontweight='bold')
    ax.set_title('Hit Rate Distribution by K\n(variability across queries)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    figures['hr_distribution'] = fig4
    
    # ========== FIGURE 5: MRR Distribution ==========
    fig5, ax = plt.subplots(figsize=(10, 6))
    data_mrr = [all_mrr[k] for k in k_values]
    bp2 = ax.boxplot(data_mrr, positions=positions, patch_artist=True, widths=0.6,
                      boxprops=dict(facecolor='#A23B72', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=11, fontweight='bold')
    ax.set_ylabel('MRR', fontsize=13, fontweight='bold')
    ax.set_title('MRR Distribution by K\n(variability across queries)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    figures['mrr_distribution'] = fig5
    
    # ========== FIGURE 6: Precision Distribution ==========
    fig6, ax = plt.subplots(figsize=(10, 6))
    data_p = [all_precision[k] for k in k_values]
    bp3 = ax.boxplot(data_p, positions=positions, patch_artist=True, widths=0.6,
                      boxprops=dict(facecolor='#F18F01', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2.5),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    ax.set_xticks(positions)
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision Distribution by K\n(variability across queries)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    figures['precision_distribution'] = fig6
    
    # ========== FIGURE 7: Heatmap ==========
    fig7, ax = plt.subplots(figsize=(10, 6))
    metrics_data = np.array([hr_values, mrr_values, p_values])
    im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'k={k}' for k in k_values], fontsize=11, fontweight='bold')
    ax.set_yticks(range(3))
    ax.set_yticklabels(['HR@K', 'MRR@K', 'P@K'], fontsize=12, fontweight='bold')
    ax.set_title('Metrics Heatmap\n(all metrics at each K)', fontsize=14, fontweight='bold')
    for i in range(3):
        for j in range(len(k_values)):
            text = ax.text(j, i, f'{metrics_data[i, j]:.2f}', ha="center", va="center", 
                          color="black", fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=11, fontweight='bold')
    plt.tight_layout()
    figures['heatmap'] = fig7
    
    # ========== FIGURE 8: Comparison Bar Chart ==========
    fig8, ax = plt.subplots(figsize=(10, 6))
    k_idx = 1  # Compare at k=5 or second value
    if len(k_values) > 1:
        metrics_names = ['Hit Rate', 'MRR', 'Precision']
        metrics_vals = [hr_values[k_idx], mrr_values[k_idx], p_values[k_idx]]
        colors_bar = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax.bar(metrics_names, metrics_vals, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title(f'Metric Comparison @ K={k_values[k_idx]}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, metrics_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    figures['comparison'] = fig8
    
    # ========== FIGURE 9: Combined Trends ==========
    fig9, ax = plt.subplots(figsize=(12, 7))
    ax2 = ax.twinx()
    
    line1 = ax.plot(k_values, [v*100 for v in hr_values], 'o-', linewidth=3, markersize=10, 
                    color='#2E86AB', label='Hit Rate (%)', alpha=0.8)
    line2 = ax2.plot(k_values, mrr_values, 's-', linewidth=3, markersize=10, 
                     color='#A23B72', label='MRR', alpha=0.8)
    line3 = ax2.plot(k_values, p_values, '^-', linewidth=3, markersize=10, 
                     color='#F18F01', label='Precision', alpha=0.8)
    
    ax.set_xlabel('K (Number of Recommendations)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hit Rate (%)', fontsize=13, fontweight='bold', color='#2E86AB')
    ax2.set_ylabel('MRR / Precision', fontsize=13, fontweight='bold', color='#A23B72')
    ax.set_title('All Metrics Trends Combined', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax.grid(True, alpha=0.3)
    
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=12)
    
    plt.tight_layout()
    figures['combined'] = fig9
    
    return figures


def save_metrics_report(metrics, results_dir="model3/evaluation_plots"):
    import os
    os.makedirs(results_dir, exist_ok=True)
    
    figure_names = {
        'hit_rate': 'HR_trend.png',
        'mrr': 'MRR_trend.png',
        'precision': 'Precision_trend.png',
        'hr_distribution': 'HR_distribution.png',
        'mrr_distribution': 'MRR_distribution.png',
        'precision_distribution': 'Precision_distribution.png',
        'heatmap': 'metrics_heatmap.png',
        'comparison': 'metric_comparison.png',
        'combined': 'all_metrics_combined.png'
    }
    
    print("\n" + "="*70)
    print("SAVING VISUALIZATIONS")
    print("="*70)
    
    for fig_key, fig_name in figure_names.items():
        fig_path = os.path.join(results_dir, fig_name)
        print(f"✓ Saved {fig_name}")
    
    csv_path = os.path.join(results_dir, "recommendation_metrics.csv")
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved recommendation_metrics.csv")
    
    print(f"\nAll visualizations saved to: {results_dir}/")
    print("="*70)


if __name__ == "__main__":
    embeddings = np.load('model3/image_embeddings.npy')
    csv_path = "data/food-ingredients-and-recipe-dataset-with-images/versions/1/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv"
    
    metrics, figures_dict = evaluate_recommendation_metrics_with_viz(embeddings, csv_path, top_k_values=[1, 5, 10, 20], num_queries=100)
    
    print("\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    for metric, value in metrics.items():
        if metric.startswith('HR'):
            print(f"{metric}: {value*100:.2f}%")
        else:
            print(f"{metric}: {value:.4f}")
    
    import os
    results_dir = "model3/evaluation_plots"
    os.makedirs(results_dir, exist_ok=True)
    
    figure_names = {
        'hit_rate': 'HR_trend.png',
        'mrr': 'MRR_trend.png',
        'precision': 'Precision_trend.png',
        'hr_distribution': 'HR_distribution.png',
        'mrr_distribution': 'MRR_distribution.png',
        'precision_distribution': 'Precision_distribution.png',
        'heatmap': 'metrics_heatmap.png',
        'comparison': 'metric_comparison.png',
        'combined': 'all_metrics_combined.png'
    }
    
    print("\n" + "="*70)
    print("SAVING VISUALIZATIONS")
    print("="*70)
    
    for fig_key, fig_name in figure_names.items():
        if fig_key in figures_dict:
            fig_path = os.path.join(results_dir, fig_name)
            figures_dict[fig_key].savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved {fig_name}")
            plt.close(figures_dict[fig_key])
    
    csv_path = os.path.join(results_dir, "recommendation_metrics.csv")
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved recommendation_metrics.csv")
    
    print(f"\nAll visualizations saved to: {os.path.abspath(results_dir)}/")
    print("="*70)
