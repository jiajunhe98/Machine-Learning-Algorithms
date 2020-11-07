import numpy as np


class KernelPCA(object):
    """
    This class implements Kernel PCA.

    Args:
        k(int): Dimension amount of transformed data.
        kernel(string): Kernel function. {"linear": linear kernel, "poly": polynomial kernel,
                                        "rbf": radial basis kernel, "sigmoid": sigmoid kernel.}
        gamma: Parameter in poly/rbf/sigmoid kernel.
        degree: Parameter in poly kernel.
        c: Constant parameter in poly/sigmoid kernel.

    Attributes:
        KernelPCA.alphas: Centered kernel matrix's eigenvector matrix, shape {k, amount_of_training_data}
        KernelPCA.lambdas: Centered kernel matrix's eigenvalue, shape {k,}

    Methods:
        KernelPCA.fit(x): Fit the model from data in x.
        KernelPCA.transform(x): Transform data in x.
        KernelPCA.fit_transform(x): Fit the model from data in x and transform x.

    """

    def __init__(self, k=None, kernel="linear", gamma=None, degree=3, c=1):
        """
        Constructor method.

        """
        self.k = k
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.c = c
        self.x = None   # training data
        self.mean = None    # Means of each dimension of training data, shape {dimension_of_training_data,}
        self.alphas = None
        self.lambdas = None
        self.centering1, self.centering2, self.centering3 = (None,) * 3    # Used for kernel matrix's centralization
    
    def fit(self, x):
        """
        Fit the model from data in x.

        Args:
            x: 2D numpy array, shape {amount_of_data, number_of_dimensions_of_each_data}

        Return:
            self: object

        """
        self.x = np.array(x)
        if self.x.ndim != 2:
            raise ValueError("Dimension of data must be 2. Use np.reshape() to reshape the data.")
        if self.k is None:
            self.k = self.x.shape[1]
        self.k = min(self.k, self.x.shape[0])
        self.mean = np.average(self.x, axis=0)
        self.x -= self.mean
        kernel = self.__kernel(self.x, self.x)
        self.centering1 = np.sum(kernel, axis=0) / self.x.shape[0]
        self.centering2 = np.reshape(np.sum(kernel, axis=1) / self.x.shape[0], [self.x.shape[0], -1])
        self.centering3 = np.sum(kernel) / self.x.shape[0] ** 2
        kernel += -self.centering1 - self.centering2 + self.centering3  # centering
        u, s, v = np.linalg.svd(kernel)
        self.lambdas, self.alphas = s[:self.k], v[:self.k]
        return self

    def transform(self, x):
        """
        Transform data in x.

        Args:
            x: 2D numpy array, shape {amount_of_data, number_of_dimensions_of_each_data}

        Return:
            Transformed data, variances are normalized.

        """
        x = np.array(x)
        kernel = self.__kernel(self.x, x - self.mean)
        kernel += self.centering3
        kernel -= self.centering2
        kernel -= self.__kernel(np.reshape(self.mean, [1, self.x.shape[1]]), x - self.mean)    # centering

        non_zeros = np.nonzero(self.lambdas)
        scaled_alphas = np.zeros_like(self.alphas)
        scaled_alphas[non_zeros] = self.alphas[non_zeros] / np.sqrt(np.reshape(self.lambdas, [-1, 1])[non_zeros])
        # Adjust the var of transformed data to 1
        res = np.dot(scaled_alphas, kernel)
        return res.T

    def fit_transform(self, x):
        """
        Fit the model from data in x and transform x.

        Args:
            x: 2D numpy array, shape {amount_of_data, number_of_dimensions_of_each_data}

        Return:
            Transformed data, variances are normalized.

        """

        self.fit(x)
        return self.transform(x)

    def __kernel(self, x, y):
        """
        Calculate kernel value between data in x and data in y.

        Return:
            Kernel , shape {amount_of_data_in_x, amount_of_data_in_y}

        """
        res = None
        if self.kernel == "linear":
            res = np.dot(x, y.T)
        if self.gamma is None:
            self.gamma = self.x.shape[1]
        if self.kernel == "poly":
            res = np.dot(x, y.T) * self.gamma + self.c
            res = res ** self.degree
        if self.kernel == "rbf":
            res = np.zeros(y.shape[0])
            for i in range(x.shape[0]):
                res = np.vstack((res, np.sum((y - x[i]) ** 2, axis=1)))
            res = np.exp(-self.gamma * res[1:])
        if self.kernel == "sigmoid":
            res = np.dot(x, y.T) * self.gamma - self.c
            res = np.tanh(res)
        return np.reshape(res, [x.shape[0], y.shape[0]])    # in case that x/y contains only one data.





import sklearn.decomposition as skd
import sklearn.datasets as sks
kpca = skd.KernelPCA(5, "sigmoid")
iris = sks.load_iris().data
kpca.fit(iris)
print(kpca.lambdas_)
print(kpca.transform(iris[:2]))
#print(kpca.alphas_[:2])
_kpca = KernelPCA(5, "sigmoid")
_kpca.fit(iris)
print(_kpca.lambdas)
print(_kpca.transform(iris[:2]))
#print(_kpca.alphas.shape)


import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kpca = KernelPCA(kernel="rbf", gamma=10)
X_kpca = kpca.fit_transform(X)


# Plot results

plt.figure()
plt.subplot()
plt.subplot(1, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')


plt.subplot(1, 2, 2, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel(r"1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")


plt.tight_layout()
plt.show()

