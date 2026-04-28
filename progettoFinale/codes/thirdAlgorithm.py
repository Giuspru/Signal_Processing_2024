import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

# ============================================================================
# NORMALIZED SPECTRAL CLUSTERING SECONDO NG, JORDAN, WEISS (2002)
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
# Step 3: Calcolo della Laplaciana Normalizzata Simmetrica L_sym
# ----------------------------------------------------------------------------
D = np.diag(np.sum(W, axis=1))
print("\nMatrice dei Gradi D:\n", D)

# Calcolo D^(-1/2) evitando divisioni per zero
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-12))

# L_sym = I - D^(-1/2) @ W @ D^(-1/2)
L_sym = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
print("\nMatrice Laplaciana Normalizzata Simmetrica L_sym:\n", L_sym)

# ----------------------------------------------------------------------------
# Step 4: Calcolo dei primi k autovettori di L_sym
# ----------------------------------------------------------------------------
k_clusters = 3

# Calcolo autovalori e autovettori di L_sym
eigVals, eigVect = np.linalg.eigh(L_sym)

# Gli autovalori sono già ordinati in ordine crescente
print("\nAutovalori di L_sym:")
print(eigVals[:10])  # Mostriamo i primi 10

# Selezioniamo i primi k autovettori (quelli con autovalori più piccoli)
U = eigVect[:, :k_clusters]
print(f"\nMatrice U degli autovettori (shape: {U.shape}):")
print(U)

# ----------------------------------------------------------------------------
# Step 5: Normalizzazione delle righe di U per creare T
# ----------------------------------------------------------------------------
# Ogni riga di T deve avere norma 1
# t_ij = u_ij / sqrt(sum_k u_ik^2)

# Calcolo delle norme delle righe
row_norms = np.linalg.norm(U, axis=1, keepdims=True)
# Evito divisioni per zero
row_norms = np.where(row_norms == 0, 1, row_norms)

# Normalizzo le righe
T = U / row_norms

print(f"\nMatrice T normalizzata (shape: {T.shape}):")
print(T)

# Verifica: le righe di T devono avere norma 1
print("\nVerifica norme delle righe di T:")
print(np.linalg.norm(T, axis=1))

# ----------------------------------------------------------------------------
# Step 6: Creazione dei punti yi dalla matrice T
# ----------------------------------------------------------------------------
# Ogni yi è la i-esima riga di T
Y = T  # Y ha dimensione (n x k)
print(f"\nPunti Y per il clustering (shape: {Y.shape}):")
print(Y)

# ----------------------------------------------------------------------------
# Step 7: K-means sui punti yi
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
axes[1].set_title("Spectral Clustering (Ng, Jordan & Weiss 2002)")
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
plt.title("Spettro degli Autovalori di L_sym")
plt.xlabel("Indice")
plt.ylabel("Autovalore λ")
plt.grid(True, alpha=0.3)
plt.axvline(x=k_clusters-0.5, color='r', linestyle='--', 
            label=f'k={k_clusters} clusters')
plt.legend()
plt.show()

# Visualizzazione 3: Confronto U vs T
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matrice U (prima della normalizzazione)
im1 = axes[0].imshow(U, aspect='auto', cmap='RdBu_r')
axes[0].set_title('Matrice U (autovettori)')
axes[0].set_xlabel('Dimensione autovettore')
axes[0].set_ylabel('Nodo')
plt.colorbar(im1, ax=axes[0])

# Matrice T (dopo la normalizzazione)
im2 = axes[1].imshow(T, aspect='auto', cmap='RdBu_r')
axes[1].set_title('Matrice T (righe normalizzate)')
axes[1].set_xlabel('Dimensione autovettore')
axes[1].set_ylabel('Nodo')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Visualizzazione 4: Embedding space (se k=2 o k=3)
if k_clusters == 2:
    plt.figure(figsize=(8, 6))
    for i in range(k_clusters):
        mask = cluster_labels == i
        plt.scatter(T[mask, 0], T[mask, 1], 
                   label=f'Cluster {i+1}', s=100, alpha=0.7)
    plt.xlabel('Componente normalizzata 1')
    plt.ylabel('Componente normalizzata 2')
    plt.title('Embedding nello spazio normalizzato (matrice T)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Aggiungiamo un cerchio unitario per mostrare che i punti sono sulla superficie
    circle = plt.Circle((0, 0), 1, fill=False, color='black', 
                       linestyle='--', linewidth=2, alpha=0.5)
    plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.show()
    
elif k_clusters == 3:
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(k_clusters):
        mask = cluster_labels == i
        ax.scatter(T[mask, 0], T[mask, 1], T[mask, 2],
                  label=f'Cluster {i+1}', s=100, alpha=0.7)
    
    ax.set_xlabel('Componente normalizzata 1')
    ax.set_ylabel('Componente normalizzata 2')
    ax.set_zlabel('Componente normalizzata 3')
    ax.set_title('Embedding 3D nello spazio normalizzato (matrice T)')
    ax.legend()
    
    # Disegno sfera unitaria
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    plt.show()

print("\n" + "="*70)
print("ALGORITMO COMPLETATO - Normalized Spectral Clustering (Ng, Jordan & Weiss)")
print("="*70)