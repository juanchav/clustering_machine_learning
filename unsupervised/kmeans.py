import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances

import numpy as np

class K_Means:

    def __init__(self,X=None,K=None,max_iters=1000):
        self.X=X
        self.K=K
        self.max_iters=max_iters
        np.random.seed(123)
    
    def fit(self,X,K,max_iters):
        self.X=X
        self.K=K
        self.max_iters=max_iters

        #gt burn first centroids      
        self.centroids = self.X[np.random.choice(len(self.X), self.K, replace=False)]
        for i in range(self.max_iters):
            # Asignación de clúster
            # miro la distancia euclidiana entre los registros (X,col) y centroides (K,col) y llevo todo
            # a una matriz de 3 dimensiones K,x,col y suma dichas distancias, generando una nueva matriz
            # distancias (k,X) que indica la distancia del punto al centroide
            distances = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            # Recomputación de centroides
            for j in range(self.K):
                self.centroids[j] = np.mean(self.X[self.labels == j], axis=0)

    def transform(self,X):
        distances = []
        for c in self.centroids:
            distances.append(np.linalg.norm(X - c, axis=1))  # Calcula la distancia entre cada punto de datos y el centroide c

        distances = np.array(distances)  # Convierte a una matriz de forma (k, n)
        return  np.argmin(distances, axis=0)  # Etiqueta cada punto de datos con el clúster correspondiente al centroide más cercano
        
    def fit_transform(self,X,K,max_iters):    
        self.fit(X,K,max_iters)
        return self.transform(X)
