import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

def generate_figure3():
    # 1. Load the real training data
    data_path = 'd:/PixelProspector-Core/02_unsupervised_ml/training_data.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    X = data['game_features']
    y = data['game_labels']
    
    # 2. Setup Springer Academic Styling
    # Muted, professional palette: Slate Gray, Steel Blue, Navy Blue
    colors = ['#708090', '#4682B4', '#000080']
    cluster_names = ['Flop', 'Niche', 'Breakout']
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Left Panel: Cluster Scatter Plot (PCA) ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    for i, color in enumerate(colors):
        mask = (y == i)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=cluster_names[i], 
                    alpha=0.8, edgecolors='none', s=40)
        
    ax1.set_title('PCA Reduction of Game Features')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend(title='Cluster', loc='best', frameon=False)
    
    # --- Right Panel: Silhouette Plot ---
    # Compute silhouette scores
    silhouette_avg = silhouette_score(X, y)
    sample_silhouette_values = silhouette_samples(X, y)
    
    y_lower = 10
    for i, color in enumerate(colors):
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax2.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.8)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, cluster_names[i], ha='right', va='center')
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
        
    ax2.set_title('Silhouette Plot for Game Clusters')
    ax2.set_xlabel('Silhouette Coefficient Values')
    ax2.set_ylabel('Cluster Label')
    
    # The vertical line for average silhouette score of all the values
    ax2.axvline(x=silhouette_avg, color="black", linestyle="--")
    ax2.text(silhouette_avg + 0.02, y_lower - 10, f"Average\n~{silhouette_avg:.4f}", 
             color="black", va='top')
    
    ax2.set_yticks([])  # Clear the yaxis labels / ticks
    ax2.set_xticks(np.arange(-0.1, 1.1, 0.2))
    
    plt.tight_layout()
    
    # Save output as high-res PNG
    output_path = 'd:/PixelProspector-Core/figure3_clusters.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved figure to {output_path}")

if __name__ == "__main__":
    generate_figure3()
