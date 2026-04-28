import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from scipy.linalg import eigh  # Importiamo eigh da scipy

# ============================================================================
# NORMALIZED SPECTRAL CLUSTERING SECONDO SHI E MALIK (2000)
# ============================================================================

# Segnale definito su Grafo 
np.random.seed(100)
X = np.random.randint(100, size=(25))
print("Segnale X:", X)
print("Numero di nodi:", len(X))

# ----------------------------------------------------------------------------
# Step 1: Creazione della Matrice di Similarità S
# ----------------------------------------------------------------------------
sigma = 60
n = len(X)
S = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        S[i, j] = np.exp(-((X[i] - X[j]) ** 2) / (2 * sigma ** 2))

print("\nMatrice di Similarità S:\n", S)

# ----------------------------------------------------------------------------
# Step 2: Costruzione del grafo di similarità (k-NN)
# ----------------------------------------------------------------------------
k_neighbors = 5
S_knn = np.zeros_like(S)

for i in range(n):
    neighbors = np.argsort(S[i])[::-1]
    count = 0
    for j in neighbors:
        if i == j:
            continue
        S_knn[i, j] = S[i, j]
        count += 1
        if count == k_neighbors:
            break

# Simmetrizzazione (grafo non diretto)
W = np.maximum(S_knn, S_knn.T)
print("\nMatrice di Adiacenza pesata W:\n", W)

# ----------------------------------------------------------------------------
# Step 3: Calcolo della Laplaciana non normalizzata L
# ----------------------------------------------------------------------------
D = np.diag(np.sum(W, axis=1))
L = D - W
print("\nMatrice dei Gradi D:\n", D)
print("\nMatrice Laplaciana L:\n", L)

# ----------------------------------------------------------------------------
# Step 4: Calcolo dei primi k autovettori generalizzati di Lu = λDu
# ----------------------------------------------------------------------------
k_clusters = 3

# Risoluzione del problema agli autovalori generalizzato: L @ u = λ @ D @ u
# Usiamo scipy.linalg.eigh che supporta problemi generalizzati
eigVals, eigVect = eigh(L, D)

# Gli autovalori sono già ordinati in ordine crescente
print("\nAutovalori del problema generalizzato Lu = λDu:")
print(eigVals[:10])  # Mostriamo i primi 10

# Selezioniamo i primi k autovettori (quelli con autovalori più piccoli)
U = eigVect[:, :k_clusters]
print(f"\nMatrice U degli autovettori (shape: {U.shape}):")
print(U)

# ----------------------------------------------------------------------------
# Step 5: Creazione dei punti yi dalla matrice U
# ----------------------------------------------------------------------------
# Ogni yi è la i-esima riga di U
Y = U  # Y ha dimensione (n x k)
print(f"\nPunti Y per il clustering (shape: {Y.shape}):")
print(Y)

# ----------------------------------------------------------------------------
# Step 6: K-means sui punti yi
# ----------------------------------------------------------------------------
def kmeans_numpy(X, k, max_iter=100, random_state=42):
    """
    Implementazione K-means da zero
    """
    np.random.seed(random_state)
    n, d = X.shape
    
    # Inizializzazione casuale dei centroidi
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]
    
    for iteration in range(max_iter):
        # Assegnazione ai cluster
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Aggiornamento centroidi
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        
        # Controllo convergenza
        if np.allclose(centroids, new_centroids):
            print(f"K-means convergenza raggiunta all'iterazione {iteration}")
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Esecuzione K-means
cluster_labels, centroids = kmeans_numpy(Y, k_clusters, random_state=42)
print("\nCluster assegnati (labels):", cluster_labels)
print("Centroidi finali:\n", centroids)

# ----------------------------------------------------------------------------
# Output: Creazione dei cluster Ai
# ----------------------------------------------------------------------------
clusters = {}
for i in range(k_clusters):
    clusters[i] = np.where(cluster_labels == i)[0].tolist()
    print(f"Cluster A{i+1}: {clusters[i]}")

# ============================================================================
# VISUALIZZAZIONI
# ============================================================================

# Visualizzazione 1: Grafo con clustering
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Grafo originale
G = nx.from_numpy_array(W)
pos = nx.spring_layout(G, seed=42)

axes[0].set_title("Grafo basato su similarità gaussiana")
nx.draw(
    G,
    pos,
    ax=axes[0],
    with_labels=True,
    node_size=300,
    node_color="lightblue",
    font_size=8
)

# Grafo con clustering
axes[1].set_title("Spectral Clustering (Shi & Malik 2000)")
nx.draw(
    G,
    pos,
    ax=axes[1],
    node_color=cluster_labels,
    cmap=plt.cm.Set1,
    with_labels=True,
    node_size=300,
    font_size=8
)

plt.tight_layout()
plt.show()

# Visualizzazione 2: Spettro degli autovalori
plt.figure(figsize=(8, 5))
plt.plot(eigVals[:15], 'o-', linewidth=2, markersize=8)
plt.title("Spettro degli Autovalori (problema generalizzato Lu = λDu)")
plt.xlabel("Indice")
plt.ylabel("Autovalore λ")
plt.grid(True, alpha=0.3)
plt.axvline(x=k_clusters-0.5, color='r', linestyle='--', 
            label=f'k={k_clusters} clusters')
plt.legend()
plt.show()

# Visualizzazione 3: Embedding space (se k=2 o k=3)
if k_clusters == 2:
    plt.figure(figsize=(8, 6))
    for i in range(k_clusters):
        mask = cluster_labels == i
        plt.scatter(Y[mask, 0], Y[mask, 1], 
                   label=f'Cluster {i+1}', s=100, alpha=0.7)
    plt.xlabel('Autovettore 1')
    plt.ylabel('Autovettore 2')
    plt.title('Embedding nello spazio degli autovettori')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
elif k_clusters == 3:
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(k_clusters):
        mask = cluster_labels == i
        ax.scatter(Y[mask, 0], Y[mask, 1], Y[mask, 2],
                  label=f'Cluster {i+1}', s=100, alpha=0.7)
    
    ax.set_xlabel('Autovettore 1')
    ax.set_ylabel('Autovettore 2')
    ax.set_zlabel('Autovettore 3')
    ax.set_title('Embedding 3D nello spazio degli autovettori')
    ax.legend()
    plt.show()

print("\n" + "="*70)
print("ALGORITMO COMPLETATO - Normalized Spectral Clustering (Shi & Malik)")
print("="*70)