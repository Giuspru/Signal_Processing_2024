import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SPECTRAL CLUSTERING - VERSIONE MIGLIORATA E COMPLETA
# ============================================================================

class SpectralClustering:
    """
    Implementazione completa di Spectral Clustering con tre varianti:
    1. Unnormalized (base)
    2. Normalized (Shi & Malik 2000)
    3. Normalized (Ng, Jordan & Weiss 2002)
    """
    
    def __init__(self, n_clusters=3, sigma=60, k_neighbors=4, random_state=42):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
        # Attributi che verranno popolati
        self.X = None
        self.S = None
        self.W = None
        self.D = None
        self.L = None
        self.L_sym = None
        self.G = None
        self.pos = None
        
    def create_signal(self, n_nodes=10):
        """Genera un segnale casuale su n nodi"""
        np.random.seed(self.random_state)
        self.X = np.random.randint(100, size=n_nodes)
        print(f"✓ Segnale generato: {n_nodes} nodi")
        print(f"  Valori: {self.X}")
        return self.X
    # Va bene!
    
    def compute_similarity_matrix(self):
        """Calcola la matrice di similarità gaussiana"""
        n = len(self.X)
        self.S = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                self.S[i, j] = np.exp(-((self.X[i] - self.X[j]) ** 2) / 
                                     (2 * self.sigma ** 2))
        
        print(f"\n✓ Matrice di similarità calcolata (σ={self.sigma})")
        return self.S
    
    # Va bene!
    
    def build_knn_graph(self):
        """Costruisce il grafo k-NN dalla matrice di similarità"""
        n = len(self.X)
        S_knn = np.zeros_like(self.S)
        
        for i in range(n):
            neighbors = np.argsort(self.S[i])[::-1]
            count = 0
            for j in neighbors:
                if i == j:
                    continue
                S_knn[i, j] = self.S[i, j]
                count += 1
                if count == self.k_neighbors:
                    break

        # Simmetrizzazione
        self.W = np.maximum(S_knn, S_knn.T)
        
        # Creazione grafo NetworkX
        self.G = nx.from_numpy_array(self.W)
        self.pos = nx.spring_layout(self.G, seed=42)
        
        n_edges = np.sum(self.W > 0) / 2
        print(f"✓ Grafo k-NN costruito (k={self.k_neighbors})")
        print(f"  Nodi: {n}, Archi: {int(n_edges)}")
        return self.W
    
    # Va bene! Controllare meglio la funzione con il ciclo.
    
    def compute_laplacians(self):
        """Calcola tutte le matrici Laplaciane"""
        # Matrice dei gradi
        self.D = np.diag(np.sum(self.W, axis=1))
        
        # Laplaciana non normalizzata
        self.L = self.D - self.W
        
        # Laplaciana normalizzata simmetrica
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(self.W, axis=1) + 1e-12))
        self.L_sym = np.eye(len(self.W)) - D_inv_sqrt @ self.W @ D_inv_sqrt
        
        print(f"\n✓ Laplaciane calcolate")
        return self.L, self.L_sym
    
    # Va bene!
    
    def fit_unnormalized(self):
        """
        Spectral Clustering non normalizzato (base)
        Usa autovettori di L direttamente
        """
        eigVals, eigVect = np.linalg.eigh(self.L)
        U = eigVect[:, :self.n_clusters]
        labels, centroids = self._kmeans(U)
        
        return {
            'method': 'Unnormalized',
            'labels': labels,
            'centroids': centroids,
            'eigvals': eigVals,
            'eigvect': U,
            'embedding': U
        }
    
    def fit_shi_malik(self):
        """
        Normalized Spectral Clustering (Shi & Malik 2000)
        Risolve il problema generalizzato: L u = λ D u
        """
        eigVals, eigVect = eigh(self.L, self.D)
        U = eigVect[:, :self.n_clusters]
        labels, centroids = self._kmeans(U)
        
        return {
            'method': 'Shi & Malik (2000)',
            'labels': labels,
            'centroids': centroids,
            'eigvals': eigVals,
            'eigvect': U,
            'embedding': U
        }
    
    def fit_ng_jordan_weiss(self):
        """
        Normalized Spectral Clustering (Ng, Jordan & Weiss 2002)
        Usa L_sym e normalizza le righe
        """
        eigVals, eigVect = np.linalg.eigh(self.L_sym)
        U = eigVect[:, :self.n_clusters]
        
        # Normalizzazione delle righe
        row_norms = np.linalg.norm(U, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1, row_norms)
        T = U / row_norms
        
        labels, centroids = self._kmeans(T)
        
        return {
            'method': 'Ng, Jordan & Weiss (2002)',
            'labels': labels,
            'centroids': centroids,
            'eigvals': eigVals,
            'eigvect': U,
            'embedding': T,
            'normalized_matrix': T
        }
    
    def _kmeans(self, X, max_iter=100):
        """K-means ottimizzato"""
        np.random.seed(self.random_state)
        n, d = X.shape
        
        # Inizializzazione k-means++
        centroids = self._kmeans_plusplus_init(X, self.n_clusters)
        
        for iteration in range(max_iter):
            distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(self.n_clusters)
            ])
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return labels, centroids
    
    def _kmeans_plusplus_init(self, X, k):
        """Inizializzazione k-means++ per migliori risultati"""
        n = X.shape[0]
        centroids = [X[np.random.randint(n)]]
        
        for _ in range(1, k):
            distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) 
                                 for x in X])
            probs = distances ** 2
            probs /= probs.sum()
            cumprobs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumprobs):
                if r < p:
                    centroids.append(X[j])
                    break
        
        return np.array(centroids)
    
    def compute_metrics(self, labels):
        """Calcola metriche di qualità del clustering"""
        # Silhouette score approssimato
        n = len(labels)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.abs(self.X[i] - self.X[j])
        
        silhouette_scores = []
        for i in range(n):
            same_cluster = labels == labels[i]
            other_clusters = labels != labels[i]
            
            if np.sum(same_cluster) > 1:
                a = np.mean(distances[i, same_cluster])
                b = np.min([np.mean(distances[i, labels == k]) 
                           for k in range(self.n_clusters) if k != labels[i]])
                s = (b - a) / max(a, b)
                silhouette_scores.append(s)
        
        return {
            'silhouette_score': np.mean(silhouette_scores) if silhouette_scores else 0,
            'n_clusters': len(np.unique(labels)),
            'cluster_sizes': [np.sum(labels == i) for i in range(self.n_clusters)]
        }


# ============================================================================
# VISUALIZZAZIONE AVANZATA
# ============================================================================

class SpectralClusteringVisualizer:
    """Classe per visualizzazioni avanzate"""
    
    def __init__(self, sc):
        self.sc = sc
        
    def plot_similarity_matrix(self):
        """Visualizza la matrice di similarità"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matrice S originale
        im1 = axes[0].imshow(self.sc.S, cmap='hot', interpolation='nearest')
        axes[0].set_title('Matrice di Similarità S')
        axes[0].set_xlabel('Nodo j')
        axes[0].set_ylabel('Nodo i')
        plt.colorbar(im1, ax=axes[0])
        
        # Matrice W (k-NN)
        im2 = axes[1].imshow(self.sc.W, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Matrice di Adiacenza W (k={self.sc.k_neighbors})')
        axes[1].set_xlabel('Nodo j')
        axes[1].set_ylabel('Nodo i')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_eigenspectrum(self, results_list):
        """Confronta gli spettri degli autovalori"""
        fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 4))
        
        if len(results_list) == 1:
            axes = [axes]
        
        for idx, result in enumerate(results_list):
            eigvals = result['eigvals'][:15]
            axes[idx].plot(eigvals, 'o-', linewidth=2, markersize=8)
            axes[idx].axvline(x=self.sc.n_clusters-0.5, color='r', 
                            linestyle='--', linewidth=2, alpha=0.7,
                            label=f'k={self.sc.n_clusters}')
            axes[idx].set_title(f"Spettro - {result['method']}")
            axes[idx].set_xlabel('Indice')
            axes[idx].set_ylabel('Autovalore λ')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_graph_clustering(self, results_list):
        """Visualizza il grafo con diversi metodi di clustering"""
        n_methods = len(results_list)
        fig, axes = plt.subplots(1, n_methods + 1, 
                                figsize=(5*(n_methods+1), 5))
        
        # Grafo originale
        axes[0].set_title('Grafo Originale')
        nx.draw(
            self.sc.G,
            self.sc.pos,
            ax=axes[0],
            with_labels=True,
            node_size=400,
            node_color='lightblue',
            font_size=8,
            font_weight='bold'
        )
        
        # Clustering per ogni metodo
        for idx, result in enumerate(results_list):
            axes[idx+1].set_title(result['method'])
            nx.draw(
                self.sc.G,
                self.sc.pos,
                ax=axes[idx+1],
                node_color=result['labels'],
                cmap=plt.cm.Set3,
                with_labels=True,
                node_size=400,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_embedding_space(self, results_list):
        """Visualizza l'embedding space per k=2 o k=3"""
        if self.sc.n_clusters == 2:
            self._plot_2d_embedding(results_list)
        elif self.sc.n_clusters == 3:
            self._plot_3d_embedding(results_list)
        else:
            print(f"Visualizzazione embedding disponibile solo per k=2 o k=3")
    
    def _plot_2d_embedding(self, results_list):
        """Embedding 2D"""
        n_methods = len(results_list)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, result in enumerate(results_list):
            embedding = result['embedding']
            labels = result['labels']
            
            for i in range(self.sc.n_clusters):
                mask = labels == i
                axes[idx].scatter(embedding[mask, 0], embedding[mask, 1],
                                label=f'Cluster {i+1}', s=100, alpha=0.7)
            
            axes[idx].set_xlabel('Componente 1')
            axes[idx].set_ylabel('Componente 2')
            axes[idx].set_title(f"{result['method']}")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
            # Cerchio unitario per metodi normalizzati
            if 'normalized_matrix' in result:
                circle = plt.Circle((0, 0), 1, fill=False, color='black',
                                  linestyle='--', linewidth=2, alpha=0.5)
                axes[idx].add_patch(circle)
                axes[idx].axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_3d_embedding(self, results_list):
        """Embedding 3D"""
        from mpl_toolkits.mplot3d import Axes3D
        
        n_methods = len(results_list)
        fig = plt.figure(figsize=(7*n_methods, 6))
        
        for idx, result in enumerate(results_list):
            ax = fig.add_subplot(1, n_methods, idx+1, projection='3d')
            embedding = result['embedding']
            labels = result['labels']
            
            for i in range(self.sc.n_clusters):
                mask = labels == i
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                         label=f'Cluster {i+1}', s=100, alpha=0.7)
            
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.set_zlabel('Componente 3')
            ax.set_title(f"{result['method']}")
            ax.legend()
            
            # Sfera unitaria per metodi normalizzati
            if 'normalized_matrix' in result:
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 20)
                x_sphere = np.outer(np.cos(u), np.sin(v))
                y_sphere = np.outer(np.sin(u), np.sin(v))
                z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                              alpha=0.1, color='gray')
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison_table(self, results_list):
        """Tabella comparativa delle metriche"""
        methods = []
        silhouettes = []
        cluster_sizes = []
        
        for result in results_list:
            metrics = self.sc.compute_metrics(result['labels'])
            methods.append(result['method'])
            silhouettes.append(metrics['silhouette_score'])
            cluster_sizes.append(metrics['cluster_sizes'])
        
        # Crea tabella
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for i, method in enumerate(methods):
            sizes_str = ', '.join([f"C{j+1}:{cluster_sizes[i][j]}" 
                                  for j in range(len(cluster_sizes[i]))])
            table_data.append([
                method,
                f"{silhouettes[i]:.3f}",
                sizes_str
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metodo', 'Silhouette Score', 'Dimensioni Cluster'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.2, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colora l'header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Confronto Metodi di Spectral Clustering', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.show()


# ============================================================================
# ESECUZIONE PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SPECTRAL CLUSTERING - ANALISI COMPLETA E COMPARATIVA")
    print("="*70)
    
    # Inizializzazione
    sc = SpectralClustering(n_clusters=2, sigma=60, k_neighbors=20, random_state=100)
    
    # Pipeline completa
    print("\n[1/4] Generazione segnale e calcolo similarità...")
    sc.create_signal(n_nodes=30)
    sc.compute_similarity_matrix()
    
    print("\n[2/4] Costruzione grafo k-NN...")
    sc.build_knn_graph()
    
    print("\n[3/4] Calcolo Laplaciane...")
    sc.compute_laplacians()
    
    print("\n[4/4] Esecuzione algoritmi di clustering...")
    results = []
    
    print("  → Unnormalized Spectral Clustering")
    results.append(sc.fit_unnormalized())
    
    print("  → Shi & Malik (2000)")
    results.append(sc.fit_shi_malik())
    
    print("  → Ng, Jordan & Weiss (2002)")
    results.append(sc.fit_ng_jordan_weiss())
    
    print("\n✓ Tutti gli algoritmi completati!")
    
    # Visualizzazioni
    print("\n" + "="*70)
    print("VISUALIZZAZIONI")
    print("="*70)
    
    viz = SpectralClusteringVisualizer(sc)
    
    print("\n→ Matrici di similarità e adiacenza")
    viz.plot_similarity_matrix()
    
    print("\n→ Spettri degli autovalori")
    viz.plot_eigenspectrum(results)
    
    print("\n→ Clustering sui grafi")
    viz.plot_graph_clustering(results)
    
    print("\n→ Embedding space")
    viz.plot_embedding_space(results)
    
    print("\n→ Tabella comparativa")
    viz.plot_comparison_table(results)
    
    print("\n" + "="*70)
    print("ANALISI COMPLETATA!")
    print("="*70)