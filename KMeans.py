import numpy as np


class KMeans:
    """
    This class implements KMeans.

    Args:
        k(int): Amount of classes.
        random_seed(int): Optional. Random seed for initialization.

    Attributes:
        KMeans.k(int): Amount of classes.
        KMeans.random_seed: Optional. Random seed for initialization.
        KMeans.means: Central value of each class.
        KMeans.dimension: Amount of dimensions of each data.

    Methods:
        KMeans.fit: Fit model to the data.
        KMeans.fit_predict: Fit model to the data, and return the class labels of data.
        KMeans.predict: Use the trained model to predict the class labels of data.

    """

    def __init__(self, k, random_seed=None):
        """
        Constructor method

        """
        self.k = k
        self.random_seed = random_seed
        self.means = None
        self.dimension = None

    def fit(self, x, max_iteration=100, terminate_condition=0.0):
        """
        Fit model to the data.

        Args:
            x: Array-like. Contains training data.
            max_iteration: Maximum times of iteration.
            terminate_condition(float): A fraction. Terminate if this value of fraction of data converge. Default 0.

        """
        self.fit_predict(x, max_iteration, terminate_condition)

    def fit_predict(self, x, max_iteration=100, terminate_condition=0.0):
        """
        Fit model to the data, and return the class labels of data.

        Args:
            x: Array-like. Contains training data.
            max_iteration: Maximum times of iteration.
            terminate_condition(float): A fraction. Terminate if this value of fraction of data converge. Default 0.

        Raises:
            ValueError("Amount of data can not be less than k")

        Return:
            Array of size[amount of data,], contains labels of data.

        """
        x = np.array(x)
        if x.ndim == 0:
            x = np.reshape(x, [1, 1])
        if x.ndim == 1:
            x = np.reshape(x, [x.shape[0], 1])
        self.dimension = x.shape[1]
        if self.k > x.shape[0]:
            raise ValueError("Amount of data can not be less than k")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.means = x[np.random.choice(x.shape[0], self.k, replace=False)]
        labels = np.zeros(x.shape[0]).astype(np.int)

        for i in range(max_iteration):
            new_labels = self.__predict(x)
            if np.sum(new_labels != labels) > terminate_condition / x.shape[0]:
                labels = new_labels
            else:
                break
            for j in range(self.k):  # Re-calculate center of each class.
                self.means[j] = np.average(x[labels == j], axis=0)

        return labels

    def predict(self, x):
        """
        Use the trained model to predict the class labels of data.

        Args:
            x: Array-like. Contains data to be predicted.

        Raises:
            ValueError("Dimension not match."): data in x have different dimension with training data.

        Return:
            Array of size[amount of data,], contains labels of data.

        """
        x = np.array(x)
        if x.ndim == 0:
            x = np.reshape(x, [1, 1])
        if x.ndim == 1:
            x = np.reshape(x, [x.shape[0], 1]) if self.dimension == 1 else np.reshape(x, [1, self.dimension])
        if x.shape[1] != self.dimension:
            raise ValueError("Dimension not match.")

        return self.__predict(x)

    def __predict(self, x):
        """
        Predict the labels for input data, by calculating square Euclidean distance.

        Args:
            x: Array-like. Contains data to be predicted.

        Return:
            Array of size[amount of data,], contains labels of data.
            
        """
        labels = np.zeros(x.shape[0]).astype(np.int)
        distance = np.full(x.shape[0], np.inf)
        for i in range(self.k):
            current_distance = np.sum((x - self.means[i]) ** 2, axis=1)
            mask = distance > current_distance
            labels[mask] = i
            distance[mask] = current_distance[mask]
        return labels


from sklearn.datasets import load_iris

iris = load_iris().data
km = KMeans(3, 10)
print(km.fit_predict(iris, 1000))

import sklearn.cluster

km = sklearn.cluster.KMeans(3, random_state=1)
print(km.fit_predict(iris))
