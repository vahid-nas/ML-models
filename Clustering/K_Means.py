

import numpy as np
import matplotlib.pyplot as plt




class KMeans:
    def __init__(self, k, max_iterations=100,threshold = 51):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.cluster_labels = None
        #threshold for finding the optimal value for K
        self.threshold = threshold
        self.error_list = []

    def _initialize_centroids(self, data):
        indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data[indices]

    def _assign_clusters(self, data):
        #calculate distance between each data with all of the centroids
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        #picks the cluster label (centroid) that is the closest (having the minimum distance)
        self.cluster_labels = np.argmin(distances, axis=1)

    def _update_centroids(self, data):
        centroids = []
        for i in range(self.k):
            mask = self.cluster_labels == i
            cluster_data = data[mask]
            if len(cluster_data) > 0:
                new_centroid = np.mean(cluster_data, axis=0)
            else:
                new_centroid = np.zeros(data.shape[1])
            centroids.append(new_centroid)
        self.centroids = np.array(centroids)

    def fit(self, data):
        self._initialize_centroids(data)
        for _ in range(self.max_iterations):
            prev_centroids = self.centroids.copy()
            self._assign_clusters(data)
            self._update_centroids(data)
            #checks wether or not the two prev_centroids and new_centroids are the same with some tolerance
            if np.allclose(prev_centroids, self.centroids):
                break

    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_error(self, data):
        possible_centroids = self.centroids[self.cluster_labels]
        error = np.mean(np.square(data - possible_centroids))
        return error

    def find_optimum_k(self,data):
        k_values = range(2, self.threshold)
        for k in range(2, self.threshold):
            self.k = k
            self.fit(data)
            error = self.calculate_error(data)
            self.error_list.append(error)

        # Plot the within-cluster sum of squares (WCSS) against K
        plt.plot(k_values, self.error_list, 'bx-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Clustering Error')
        plt.title('Elbow Method')
        plt.show()