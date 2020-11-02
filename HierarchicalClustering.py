import numpy as np
from Distances import distances
import math

class HierarchicalClustering:
    """
    This class implements Hierarchical clustering.

    Args:
        x: Numpy array. Each row represents one data.
        k: Number of clusters.
        distance: {"E": Euclidean Distance, "Minkowski": Minkowski Distance, "Manhattan": Manhattan Distance,
                "C": Chebyshev Distance, "Mahalanobis": Mahalanobis Distance, "Corr": Correlation Coefficient,
                "cos": cosine}
        linkage: {"min": single linkage, "max": complete linkage, "central": central distance, "ave": average distance}
        p: If Minkowski distance is chosen, p should be set, default None.

    Attributes:
        x: Numpy array.
        distance: Method of calculating distance.
        p: p for Minkowski distance.
        k: Number of clusters.
        linkage: Method of linkage 2 classes(distances between 2 classes).
        dist: Distances matrix, contains distances between all classes.
        original_dist: Distances matrix, contains distances between all data.
        classes: List of classes. Each elements is a list of index of data in this class.

    Methods:
        get_labels: Return labels for all data.

    Raises:
        ValueError("Variance of data is 0. "): If "Corr" is chosen for distance, this error will be raised if variance of a sample is 0.

    """

    def __init__(self, x, k = 1, distance = "E", linkage = "min", p = None):
        """
        Constructor method
        """
        if x.ndim == 1: x = np.reshape(x, [1, x.shape[0]])
        if x.ndim == 0: x = np.reshape(x, [1, 1])
        self.x = x
        self.distance = distance
        self.p = p
        self.k = k
        self.linkage = linkage
        self.dist = distances(self.x, self.distance, self.p) if self.distance not in ["Corr", "cos"] \
            else 1 - distances(self.x, self.distance, self.p)
        if np.isnan(np.sum(self.dist)): raise ValueError("Variance of data is 0. ")
        for i in range(self.dist.shape[0]): self.dist[i, i] = math.inf
        self.original_dist = self.dist.copy()
        self.classes = [[i] for i in range(self.dist.shape[0])]
        self.__cluster()

    def get_labels(self):
        """
        Return labels of all data.

        Return:
            numpy array of labels.
        """
        labels = np.zeros(self.original_dist.shape[0])
        for i in range(len(self.classes)):
            labels[self.classes[i]] = i
        return labels

    def __cluster(self):
        """
        Cluster iteratively.
        """
        while len(self.classes) > self.k:
            smallest_index = np.where(self.dist == np.min(self.dist))                      # find the index for smallest value
            self.__new_distances(smallest_index[0][0], smallest_index[1][0], self.linkage)
            self.classes[smallest_index[0][0]].extend(self.classes[smallest_index[1][0]])  # merge two classes
            self.classes.pop(smallest_index[1][0])

    def __new_distances(self, i, j, linkage):
        """
        Calculate new distances between the changed class and other classes
        """
        if linkage == "max":
            self.dist[i, :] = np.max(self.dist[[i, j], :], axis = 0)
            self.dist[:, i] = np.max(self.dist[:, [i, j]], axis = 1)
        if linkage == "min":
            self.dist[i, :] = np.min(self.dist[[i, j], :], axis = 0)
            self.dist[:, i] = np.min(self.dist[:, [i, j]], axis = 1)
        if linkage == "ave":
            for k in range(self.dist.shape[0]):
                self.dist[i, k] = np.average(self.original_dist[self.classes[i], self.classes[k]])
                self.dist[k, i] = np.average(self.original_dist[self.classes[i], self.classes[k]])
        self.dist[i, i] = math.inf
        self.dist = np.delete(self.dist, j, axis=1)
        self.dist = np.delete(self.dist, j, axis=0)
        if linkage == "central":
            new_x = np.array([np.average(self.x[i],axis=0) for i in self.classes])
            self.dist = distances(new_x, self.distance, self.p) if self.distance not in ["Corr", "cos"] \
                else 1 - distances(new_x, self.distance, self.p)
            for i in range(self.dist.shape[0]): self.dist[i, i] = math.inf







a = np.array([[1,2,2,2],[1,2,1,2],[-1,-1,-2,-3],[-2,-1,-3,-2],[-3,-2,5,6],[-5,-3,6,7]])



m = HierarchicalClustering(a, 3, "cos", "min")
print(m.get_labels())
print(m.classes)







