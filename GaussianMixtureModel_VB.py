import numpy as np

class GMM(object):
    """
    This class implements GMM model by variational inference.

    Args:
        n_clusters(int): Number of Gaussian Models in GMM.

    Attributes:
        n_cluster(int): Number pf Gaussian Models in GMM.
        pz: Array-like, prior distribution of Gaussian Models in log-scale.
        qz: Array-like, posterior probabilities of each training data, i.e. q(z|x).
        means: Array-like, means of Gaussian Models.
        sigmas: List of cor-variances pf Gaussian Models.
        dim(int): Dimension of data.

    Methods:
        fit(data, max_iter): Training by data.
        predict(x): Predict the classes of input x.
    """

    def __init__(self, n_clusters):
        """Constructor method"""

        self.n_cluster = n_clusters
        self.pz = np.array([np.log(1/n_clusters)] * n_clusters)  # p(z) prior distribution, log-scale
        self.qz = None   # q(z|x) approximated posterior distribution
        self.means = None   # means of Gaussian distributions
        self.sigmas = None  # covariance matrices of Gaussian distributions
        self.dim = None

    def fit(self, data, max_iter=100):
        """
        Training by data.

        Args:
            data: Array-like, shape (n_samples, dim)
            max_iter(int): Maximum iteration times, default 100.

        Return: Model.

        """
        self.__initialization(data)
        for i in range(max_iter):
            log_probs = [self.__gaussian(data, self.means[model], self.sigmas[model]) + self.pz[model]
                         for model in range(self.n_cluster)]
            log_probs = np.vstack(log_probs) # log_probs without normalization
            self.qz = np.exp(log_probs - np.max(log_probs, axis=0))
            self.qz = self.qz / np.sum(self.qz, axis=0)
            classes = np.argmax(log_probs, axis=0)
            for model in range(self.n_cluster):
                self.means[model] = np.mean(data[classes == model, :], axis=0)
                self.sigmas[model] = np.cov(data[classes == model, :].T)
        return self

    def predict(self, x):
        """
        Predict the classes of input x.

        Args:
            x: Array-like, data to be predicted. Shape (n_samples, dim).

        Returns:
            Array-like, classes of each input data in x. Shape (n_samples,).
        """
        log_probs = [self.__gaussian(x, self.means[model], self.sigmas[model]) + self.pz[model]
                     for model in range(self.n_cluster)]
        log_probs = np.vstack(log_probs)
        classes = np.argmax(log_probs, axis=0)
        return classes

    def __initialization(self, data):
        """Initialization."""

        self.dim = data.shape[1]
        self.means = [(np.max(data, axis=0) - np.min(data, axis=0)) / self.n_cluster * (i + 1) + np.min(data, axis=0) for i in range(self.n_cluster)]
        self.means = np.vstack(self.means)
        self.sigmas = [np.eye(self.dim) for i in range(self.n_cluster)]

    def __gaussian(self, x, mean, sigma):
        """return log-scale"""

        sigma = sigma.copy() + 1e-7
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)
        log_probs = -0.5 * np.log(2 ** self.dim * np.pi ** self.dim * det_sigma) - 0.5 * np.diag(np.dot(np.dot((x - mean), inv_sigma), (x - mean).T))
        return log_probs





# test
x1 = np.random.multivariate_normal([0, 0, 0], [[1, 0, 0],[0, 1, 0], [0, 0, 2]], 100)
x2 = np.random.multivariate_normal([-4, -4, -4], [[2, 1, 1],[1, 2, 1],[1, 1, 3]], 200)
x3 = np.random.multivariate_normal([4, 2, 3], [[2, 0, 0],[0, 3, 0], [0, 0, 2]], 100)
x = np.vstack((x1,x2,x3))
gm = GMM(3)
gm.fit(x)

print(gm.qz)


predict = gm.predict(x)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[predict == 0,0], x[predict == 0,1], x[predict == 0,2])
ax.scatter(x[predict == 1,0], x[predict == 1,1], x[predict == 1,2])
ax.scatter(x[predict == 2,0], x[predict == 2,1], x[predict == 2,2])
plt.show()

print(predict)



