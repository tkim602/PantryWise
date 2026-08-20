import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import time
import umap

def evaluate_embedding_space_quality(embeddings):
    """
    Analyzes the statistical properties of learned embeddings.
    Checks: normalization, coverage, and clustering tendency.
    Fast: O(n) - runs on full dataset.
    """
    print("\n" + "="*70)
    print("METHOD 1: EMBEDDING SPACE QUALITY ANALYSIS")
    print("="*70)
    
    # l2 norm analysis
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nL2 Norm Statistics:")
    print(f"  Mean:   {norms.mean():.4f}")
    print(f"  Std:    {norms.std():.4f}")
    print(f"  Min:    {norms.min():.4f}")
    print(f"  Max:    {norms.max():.4f}")
    print(f"  Status: {'✓ Well-normalized' if 40 < norms.mean() < 50 else '⚠ Consider L2 normalization'}")
    
    # value range
    print(f"\nValue Range:")
    print(f"  Min value: {embeddings.min():.6f}")
    print(f"  Max value: {embeddings.max():.6f}")
    print(f"  Span:      {embeddings.max() - embeddings.min():.4f}")
    
    # sparsity
    zero_ratio = (embeddings == 0).sum() / embeddings.size
    print(f"\nSparsity:")
    print(f"  Zero values: {zero_ratio*100:.2f}%")
    print(f"  Status: {'✓ Dense embeddings' if zero_ratio < 0.01 else '⚠ Sparse embeddings'}")
    
    dim_means = np.abs(embeddings).mean(axis=0)
    used_dims = (dim_means > 1e-6).sum()
    print(f"\nDimensionality Usage:")
    print(f"  Total dimensions: {embeddings.shape[1]}")
    print(f"  Active dimensions: {used_dims}")
    print(f"  Usage ratio: {used_dims/embeddings.shape[1]*100:.1f}%")
    print(f"  Status: {'✓ Good' if used_dims > embeddings.shape[1]*0.8 else '⚠ Many unused dimensions'}")
    
    print("="*70)
    return {'norms': norms, 'zero_ratio': zero_ratio, 'used_dims': used_dims}

def evaluate_recommendation_quality(embeddings, sample_size=500):
    print("\n" + "="*70)
    print("METHOD 2: RECOMMENDATION QUALITY (Neighbor Consistency)")
    print("="*70)
    
    if len(embeddings) > sample_size:
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
    else:
        sample_embeddings = embeddings
    
    print(f"\nEvaluating on {len(sample_embeddings)} samples...")
    
    similarities = cosine_similarity(sample_embeddings)
    
    # for each img check rank of nearest neighbors
    mean_ranks = []
    for i in range(len(similarities)):
        sims = similarities[i].copy()
        sims[i] = -np.inf
        sorted_indices = np.argsort(-sims)  # descending order
        
        for k in [1, 5, 10]:
            if k < len(sorted_indices):
                mean_ranks.append(sorted_indices[:k].mean())
    
    reciprocal_count = 0
    total_pairs = 0
    for i in range(min(100, len(sample_embeddings))):
        top_5_i = np.argsort(-similarities[i])[:6][1:]
        for j in top_5_i:
            total_pairs += 1
            top_5_j = np.argsort(-similarities[j])[:6][1:]
            if i in top_5_j:
                reciprocal_count += 1
    
    reciprocal_ratio = reciprocal_count / total_pairs if total_pairs > 0 else 0
    
    off_diag_mask = ~np.eye(len(similarities), dtype=bool)
    inter_sims = similarities[off_diag_mask]
    
    print(f"\nNeighbor Consistency Metrics:")
    print(f"  Mean inter-image similarity: {inter_sims.mean():.4f}")
    print(f"  Std inter-image similarity:  {inter_sims.std():.4f}")
    print(f"  Reciprocal neighbor ratio:   {reciprocal_ratio*100:.1f}%")
    print(f"  Status: {'✓ Good' if reciprocal_ratio > 0.4 else '⚠ Weak'} ({reciprocal_ratio:.2%} reciprocal)")
    
    # Top-k diversity
    print(f"\nTop-K Similarity Percentiles:")
    for k in [1, 5, 10]:
        percentiles = np.percentile(inter_sims, [25, 50, 75, 90])
        print(f"  P25: {percentiles[0]:.4f}, P50: {percentiles[1]:.4f}, P75: {percentiles[2]:.4f}, P90: {percentiles[3]:.4f}")
    
    print("="*70)
    return {'reciprocal_ratio': reciprocal_ratio, 'inter_sims': inter_sims}

def visualize_with_umap(embeddings, num_samples=1000):
    """
    UMAP visualization of embedding space (optimized for speed).
    Shows global structure and clusters.
    Good for: Identifying major food categories and outliers.
    Fast: ~15-30 seconds for 1000 samples with optimized params.
    """
    print("\n" + "="*70)
    print("METHOD 3: UMAP VISUALIZATION")
    print("="*70)
    
    if len(embeddings) > num_samples:
        print(f"Sampling {num_samples} embeddings from {len(embeddings)}...")
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
    
    print(f"Running UMAP on {len(sample_embeddings)} samples (fast mode)...")
    print("  Computing neighbors...")
    
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=10,
        min_dist=0.05,
        metric='cosine',
        low_memory=False,
        verbose=False
    )
    
    print("  Fitting UMAP...")
    reduced = reducer.fit_transform(sample_embeddings)
    print("  Rendering plot...")
    
    # Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=20, c=np.arange(len(reduced)), cmap='viridis')
    plt.colorbar(label='Sample Index')
    plt.title(f"UMAP: Embedding Space Structure ({len(sample_embeddings)} samples)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()
    
    print("✓ UMAP visualization complete")
    print("="*70)
    return reduced


def evaluate_neighborhood_stability(embeddings, num_samples=1000, k=10):
    print("\n" + "="*70)
    print("METHOD 4: NEIGHBORHOOD STABILITY (Robustness)")
    print("="*70)
    
    if len(embeddings) > num_samples:
        sample_indices = np.random.choice(len(embeddings), num_samples, replace=False)
        sample_embeddings = embeddings[sample_indices]
    else:
        sample_embeddings = embeddings
    
    print(f"\nTesting neighborhood stability on {len(sample_embeddings)} samples...")
    
    # Original similarities
    sims_original = cosine_similarity(sample_embeddings)
    
    stability_scores = []
    noise_levels = [0.01, 0.05, 0.1]
    
    for noise_level in noise_levels:
        # Add small Gaussian noise
        noise = np.random.normal(0, noise_level, sample_embeddings.shape)
        perturbed_embeddings = sample_embeddings + noise
        
        # Normalize to keep on similar scale
        perturbed_embeddings = perturbed_embeddings / np.linalg.norm(perturbed_embeddings, axis=1, keepdims=True)
        
        sims_perturbed = cosine_similarity(perturbed_embeddings)
        
        # Check how many neighbors remain in top-k
        stable_neighbors = 0
        total_checks = 0
        
        for i in range(len(sample_embeddings)):
            top_k_orig = set(np.argsort(-sims_original[i])[1:k+1])  # exclude self
            top_k_pert = set(np.argsort(-sims_perturbed[i])[1:k+1])
            overlap = len(top_k_orig & top_k_pert)
            stable_neighbors += overlap
            total_checks += k
        
        stability = stable_neighbors / total_checks
        stability_scores.append(stability)
        print(f"  Noise level {noise_level:.2f}: {stability*100:.1f}% neighbors remain stable")
    
    avg_stability = np.mean(stability_scores)
    print(f"\nAverage Stability: {avg_stability*100:.1f}%")
    print(f"Status: {'✓ Robust' if avg_stability > 0.7 else '⚠ Fragile'} embeddings")
    print("="*70)
    return stability_scores


# HR@K, MRR, P@K
def evaluate_recommendation_metrics(embeddings, csv_path, top_k_values=[1, 5, 10], num_queries=100):
    print("\n" + "="*70)
    print("METHOD 5: RECOMMENDATION METRICS (HR@K, MRR@K, P@K)")
    print("="*70)
    
    # Load CSV to get ingredient info
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Sample test queries
    all_indices = np.arange(len(embeddings))
    if len(embeddings) > num_queries:
        query_indices = np.random.choice(all_indices, size=num_queries, replace=False)
    else:
        query_indices = all_indices
    
    print(f"\nTesting on {len(query_indices)} random recipes...\n")
    
    print("  Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    results = {}
    
    for k in top_k_values:
        print(f"  Computing metrics @k={k}...")
        
        # HR@K
        hits = 0
        mrr_scores = []
        precisions = []
        
        for query_idx in query_indices:
            # top-k recommendations
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
            
            if relevant_found:
                hits += 1
            mrr_scores.append(reciprocal_rank)
            precisions.append(relevant_count / k)
        
        hr_k = hits / len(query_indices)
        mrr_k = np.mean(mrr_scores)
        p_k = np.mean(precisions)
        
        results[f'HR@{k}'] = hr_k
        results[f'MRR@{k}'] = mrr_k
        results[f'P@{k}'] = p_k
        
        print(f"    HR@{k}:  {hr_k*100:6.2f}%  | MRR@{k}: {mrr_k:.4f}  | P@{k}:  {p_k:.4f}")
    
    print("\n" + "="*70)
    print("Interpretation:")
    print("  HR@K:  Hit Rate - % of queries with ≥1 relevant recommendation")
    print("  MRR@K: Mean Reciprocal Rank - how early is first relevant item?")
    print("  P@K:   Precision@K - % of recommendations that are relevant")
    print("="*70)
    
    return results




def main():
    ENABLE_UMAP = False
    
    start_time = time.time()
    
    print("\n" + "█"*70)
    print("CNN IMAGE EMBEDDING EVALUATION (5 Methods)")
    print("█"*70)
    
    print("\nLoading embeddings...")
    embeddings = np.load('model3/image_embeddings.npy')
    print(f"✓ Loaded {embeddings.shape[0]:,} embeddings (dim={embeddings.shape[1]})")
    
    csv_path = "data/food-ingredients-and-recipe-dataset-with-images/versions/1/Food Ingredients and Recipe Dataset with Image Name Mapping.normalized.csv"
    
    # Space Quality
    print("\n[1/5] Analyzing embedding space quality...")
    method1_results = evaluate_embedding_space_quality(embeddings)
    
    # Recommendation Quality (neighbor consistency)
    print("\n[2/5] Evaluating recommendation quality...")
    method2_results = evaluate_recommendation_quality(embeddings, sample_size=500)
    
    # UMAP Visualization, might acc use
    if ENABLE_UMAP:
        print("\n[3/5] Running UMAP visualization...")
        method3_results = visualize_with_umap(embeddings, num_samples=1000)
        method_count = 5
    else:
        print("\n[3/5] Skipping UMAP visualization (ENABLE_UMAP=False)")
        method3_results = None
        method_count = 5
    
    print(f"\n[4/5] Testing neighborhood stability...")
    method4_results = evaluate_neighborhood_stability(embeddings, num_samples=1000, k=10)
    
    print(f"\n[5/5] Evaluating recommendation metrics...")
    method5_results = evaluate_recommendation_metrics(embeddings, csv_path, top_k_values=[1, 5, 10], num_queries=100)
    
    total_time = time.time() - start_time
    print("\n" + "█"*70)
    print("EVALUATION COMPLETE")
    print("█"*70)
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print("\nKey Findings:")
    print(f"  • Embedding dimensionality usage: {method1_results['used_dims']}/{embeddings.shape[1]} dims")
    print(f"  • Reciprocal neighbor ratio: {method2_results['reciprocal_ratio']*100:.1f}%")
    print(f"  • Average neighborhood stability: {np.mean(method4_results)*100:.1f}%")
    print(f"\nRecommendation Performance (Method 5):")
    for metric, value in method5_results.items():
        if metric.startswith('HR'):
            print(f"  • {metric}: {value*100:.2f}%")
        else:
            print(f"  • {metric}: {value:.4f}")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()