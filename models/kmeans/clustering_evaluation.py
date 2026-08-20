import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

class ClusteringEvaluator:
    def __init__(self, tfidf_matrix, clusters=None, kmeans_model=None):
        self.tfidf_matrix = tfidf_matrix
        self.clusters = clusters
        self.kmeans_model = kmeans_model
        
    def calculate_metrics(self, sample_size=5000):
        sample_size = min(sample_size, self.tfidf_matrix.shape[0])
        sample_indices = np.random.choice(self.tfidf_matrix.shape[0], sample_size, replace=False)
        
        tfidf_sample = self.tfidf_matrix[sample_indices].toarray()
        clusters_sample = self.clusters[sample_indices]
        
        # 1. Silhouette Score
        silhouette_avg = silhouette_score(tfidf_sample, clusters_sample)
        
        # 2. Davies-Bouldin Index
        db_index = davies_bouldin_score(tfidf_sample, clusters_sample)
        
        # 3. Within-Cluster Sum of Squares (WCSS)
        wcss = self.kmeans_model.inertia_
        
        # 4. Additional metrics
        n_clusters = len(np.unique(self.clusters))
        cluster_sizes = np.bincount(self.clusters)
        
        metrics = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_index': db_index,
            'wcss': wcss,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
        }
        
        self._print_metrics(metrics)
        return metrics
    
    def _print_metrics(self, metrics):
        print("\n" + "="*50)
        print("CLUSTERING EVALUATION METRICS")
        print("="*50)
        print(f"Silhouette Score:      {metrics['silhouette_score']:.3f}")
        print(f"Davies-Bouldin Index:  {metrics['davies_bouldin_index']:.3f}")
        print(f"WCSS:                  {metrics['wcss']:.0f}")
        print(f"Number of Clusters:    {metrics['n_clusters']}")
        print(f"Average Cluster Size:  {metrics['avg_cluster_size']:.1f}")
        print(f"Min Cluster Size:      {metrics['min_cluster_size']}")
        print(f"Max Cluster Size:      {metrics['max_cluster_size']}")
        print("="*50)
    
    def elbow_analysis(self, max_k=30, sample_size=5000):
        k_range = range(2, max_k + 1)
        wcss_values = []
        silhouette_scores = []
        db_indices = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.tfidf_matrix)
            
            # WCSS
            wcss_values.append(kmeans.inertia_)
            
            sample_size_actual = min(sample_size, self.tfidf_matrix.shape[0])
            sample_indices = np.random.choice(self.tfidf_matrix.shape[0], sample_size_actual, replace=False)
            
            # Silhouette Score
            silhouette_val = silhouette_score(
                self.tfidf_matrix[sample_indices].toarray(), 
                clusters[sample_indices]
            )
            silhouette_scores.append(silhouette_val)
            
            # Davies-Bouldin Index
            db_val = davies_bouldin_score(
                self.tfidf_matrix[sample_indices].toarray(), 
                clusters[sample_indices]
            )
            db_indices.append(db_val)
            
            print(f"  k={k}: WCSS={kmeans.inertia_:.0f}, Silhouette={silhouette_val:.3f}, DB={db_val:.3f}")
        
        results = {
            'k_range': list(k_range),
            'wcss_values': wcss_values,
            'silhouette_scores': silhouette_scores,
            'db_indices': db_indices
        }
        return results
    
    def plot_analysis(self, elbow_results, current_k=None):
        k_range = elbow_results['k_range']
        wcss_values = elbow_results['wcss_values']
        silhouette_scores = elbow_results['silhouette_scores']
        db_indices = elbow_results['db_indices']
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # WCSS Elbow Plot
        ax1.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within-Cluster Sum of Squares')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        if current_k:
            ax2.axvline(x=current_k, color='red', linestyle='--', alpha=0.7)
        
        # Silhouette Score Plot
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        if current_k:
            ax2.axvline(x=current_k, color='red', linestyle='--', alpha=0.7)
        
        # Davies-Bouldin Index Plot
        ax3.plot(k_range, db_indices, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Davies-Bouldin Index')
        ax3.set_title('Davies-Bouldin Analysis')
        ax3.grid(True, alpha=0.3)
        if current_k:
            ax2.axvline(x=current_k, color='red', linestyle='--', alpha=0.7)
        
        plt.savefig('clustering_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        return fig