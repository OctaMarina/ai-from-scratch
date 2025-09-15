import numpy as np

class KMeans:
    def __init__(self, k:int, max_iter:int = 300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    @staticmethod
    def _euclidian_distance(x:np.ndarray,y:np.ndarray):
        """
        Calculate the euclidian distance between vector x and vector y
        """
        return np.sqrt(np.sum(np.square(x-y)))
    
    def compute_labels(self, x:np.ndarray):
        """
        Compute the labels for the actual values of the centroids
        """
        self.labels = []
        for point in x:
            minim = self._euclidian_distance(point,self.centroids[0])
            minim_index = 0
            for j in range(1,len(self.centroids)):
                distance = self._euclidian_distance(point,self.centroids[j])
                if distance < minim:
                    minim = distance
                    minim_index = j
            self.labels.append(minim_index)

        self.labels = np.array(self.labels)
    
    def fit(self, X: np.ndarray):
        """
        Compute k-means clustering.
        Learn the cluster centers from the data X.
        """

        # Initialize random clusters
        indices = np.random.choice(X.shape[0], size = self.k, replace=False)
        self.centroids = X[indices,:]

        last_epoch_centroids = self.centroids.copy()
        for _ in range(self.max_iter):
            self.compute_labels(X)

            for index in range(self.k):
                epoch = X[self.labels == index]
                new_center = np.mean(epoch, axis=0)
                self.centroids[index] = new_center
            if np.allclose(self.centroids, last_epoch_centroids):
                break

            last_epoch_centroids = self.centroids.copy()
        
        self.compute_labels(X)

        return self
                
    def predict(self, X: np.ndarray):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        labels = []

        for point in X:
            minim = self._euclidian_distance(point,self.centroids[0])
            minim_index = 0
            for index in range(1,len(self.centroids)):
                distance = self._euclidian_distance(point,self.centroids[index])
                if minim > distance:
                    minim = distance
                    minim_index = index
            labels.append(minim_index)
        return np.array(labels)
    
    def transform(self, X: np.ndarray):
        """
        Transform X to a cluster-distance space.
        Return an array of shape (n_samples, k) with distances
        from each point to each cluster center.
        """
        distances = []
        for point in X:
            aux_list = []
            for cluster in self.centroids:
                aux_list.append(self._euclidian_distance(point,cluster))
            distances.append(aux_list)
        return np.array(distances)
