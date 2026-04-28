import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt



#Segnale definito su Grafo 
np.random.seed(100)
X=np.random.randint(100, size=(25))
print(X)
print(len(X))

#Creazione della Matrice di Similarità 

sigma = 60
S = np.zeros((len(X) , len(X)))
print(S)


for i in range(len(X)):
    for j in range(len(X)):
        S[i,j] = np.exp(-((X[i] - X[j]) ** 2) / (2 * sigma ** 2))

print(S)

#Creazione Matrice Adiacenza: i nodi si collegano solo ai K vicini piu simili.
k = 5
n = len(X)

# 2. costruiamo matrice k-NN
S_knn = np.zeros_like(S)

for i in range(n):
    # ordiniamo gli indici per similarità decrescente
    neighbors = np.argsort(S[i])[::-1]
    
    count = 0
    for j in neighbors:
        if i == j:
            continue
        S_knn[i, j] = S[i, j]
        count += 1
        if count == k:
            break

# 3. simmetrizzazione (grafo non diretto)
W = np.maximum(S_knn, S_knn.T)


print("Matrice W:\n", W)


# Creazione grafo dalla matrice di adiacenza W
G = nx.from_numpy_array(W)

# Etichette dei nodi (valori del segnale)
labels = {i: int(X[i]) for i in range(len(X))}

# Layout del grafo
pos = nx.spring_layout(G, seed=42)

# Spessori degli archi proporzionali ai pesi
edges = G.edges(data=True)
#weights = [d['weight'] * 5 for (_, _, d) in edges]

# Disegno nodi e archi
plt.figure(figsize=(6, 4))
nx.draw(
    G,
    pos,
    with_labels=False,
    labels=labels,
    node_size=90,
    node_color="lightblue",
    #width=weights
)

# Etichette dei pesi sugli archi
#edge_labels = {(i, j): f"{d['weight']:.2f}" for i, j, d in edges}
#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Graph basato su similarità gaussiana")
plt.show()



#Creazione Matrince dei Gradi: 
D = np.diag(np.sum(W,axis=1))
print("Matrice Gradi:\n", D)


#Laplaciana: 
L = D - W
print("Matrice Laplaciana:\n", L)

#Laplaciana Normalizzata: 
#D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-12))
#L_N = D_inv_sqrt @ W @ D_inv_sqrt
#print("Laplaciana normalizzata:\n", L_N)

# evita divisioni per zero
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-12))
L_N = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
print("Laplaciana normalizzata:\n", L_N)

#Calcolo Autovalori e Autovettori di L: 
eigVals , eigVect = np.linalg.eigh(L)
print("Autovalori L:\n", eigVals)
print("Autovettore 1:\n", eigVect[1] )

eigVals_sorted = np.sort(eigVals)

plt.figure(figsize=(6,4))
plt.plot(eigVals_sorted, 'o-')
plt.title("Spettro della Laplaciana")
plt.xlabel("Indice")
plt.ylabel("Autovalore")
plt.grid(True)
plt.show()

#Calcolo Autovalori e Autovettori di L_N:
eigVals_N, eigVect_N = np.linalg.eigh(L_N)
print("Autovalori L_N:\n", eigVals_N)
print("Autovettore 1:\n", eigVect_N[1] )

eigVals_N_sorted = np.sort(eigVals_N)

plt.figure(figsize=(6,4))
plt.plot(eigVals_N_sorted, 'o-')
plt.title("Spettro della Laplaciana")
plt.xlabel("Indice")
plt.ylabel("Autovalore")
plt.grid(True)
plt.show()




c = 4
# prendiamo i c più piccoli autovettori
U = eigVect[:, :c]
print("Matrice degli autovettori (embedding):\n", U)
print("Dimensioni Matrice U:\n", U.shape)
Y = U.T
print("Matrice degli autovettori Trasposta (embedding):\n", Y)
print("Dimensioni Matrice Y:\n", Y.shape)

# U: (n x k) autovettori
Y = U  # embedding corretto per clustering

k_clusters = 4 # scegli numero cluster


def kmeans_numpy(X, k, max_iter=100):
    n, d = X.shape
    
    # 1. inizializzazione casuale dei centroidi
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]
    
    for _ in range(max_iter):
        # 2. assegnazione cluster
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 3. aggiornamento centroidi
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        
        # 4. convergenza
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids



labels, centroids = kmeans_numpy(Y, k_clusters)
print("Cluster assegnati:", labels)

#Visualizzazione su Grafo:
G = nx.from_numpy_array(W)
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    node_color=labels,
    cmap=plt.cm.Set1,
    with_labels=False,
    node_size=90
)

plt.title("Spectral Clustering (K-means da zero)")
plt.show()