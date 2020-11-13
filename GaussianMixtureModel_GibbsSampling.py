import numpy as np
import KMeans
from scipy.stats import invwishart, multivariate_normal


class GaussianMixtureModel(object):
    """
    This class implements Gaussian Mixture Model(GMM), using Gibbs Sampling algorithm to estimate parameters.

    Args:
        n_cluster(int): number of clusters.
        max_iteration(int): maximum number of iterations of Gibbs sampling, default 100.
        precision_ratio(float): terminate iteration if labels' changing percentage is less than it. Default 0.01.

    Attributes:
        sigmas_: Array-like. Co-var matrix of the model. Shape {n_cluster, dimension, dimension}
        mus_: Array-like. Means of the models. Shape {n_cluster, dimension}
        alphas_: Array-like. Ratio of different gaussian distribution in the model. Shape {n_cluster,}

    Methods:
        fit(data, alphas=None, means=None, lambdas=None, gammas=None, psis=None): Fit the model to training data.
        predict(self, x): Predict labels for input data in x.

    """

    def __init__(self, n_cluster=3, max_iteration=100, precision_ratio=0.01):
        """
        Constructor method
        """
        self.n_clusters = n_cluster
        self.max_iteration = max_iteration
        self.precision_ratio = precision_ratio
        self.data = None    # training data
        self.dimension = None   # dimension of data
        self.n_samples = None   # size of training dataset
        self.alphas = None  # prior parameter of Dirichlet distribution
        self.means = None   # prior parameter of Normal-inverse-Wishart distribution
        self.lambdas = None    # prior parameter of Normal-inverse-Wishart distribution
        self.gammas = None    # prior parameter of Normal-inverse-Wishart distribution
        self.psis = None    # prior parameter of Normal-inverse-Wishart distribution
        self.labels = None    # predicted labels of training data
        self.sigmas_ = None    # co-var matrix of the model
        self.mus_ = None    # means of the models
        self.alphas_ = None    # ratio of different gaussian distribution in the model

    def fit(self, data, alphas=None, means=None, lambdas=None, gammas=None, psis=None):
        """
        Fit the model to training data.

        Args:
            data: Array-like. Training set. Shape {n_samples, dimension}
            alphas: Array-like. Prior parameter of Dirichlet distribution. Shape {n_cluster, }
            means: Array-like. Prior parameter of Normal-inverse-Wishart distribution. Shape {n_cluster, dimension}
            lambdas: Array-like. Prior parameter of Normal-inverse-Wishart distribution. Shape {n_cluster, }
            gammas: Array-like. Prior parameter of Normal-inverse-Wishart distribution. Shape {n_cluster, }
            psis: Array-like. Prior parameter of Normal-inverse-Wishart distribution.
                  Shape {n_cluster, dimension, dimension}

        Return:
            Model fitted to training data.

        """
        assert data.ndim == 2
        self.data = data
        self.dimension = data.shape[1]
        self.n_samples = data.shape[0]
        self.alphas, self.means, self.lambdas, self.gammas, self.psis = alphas, means, lambdas, gammas, psis
        self.__initialization()

        for i in range(self.max_iteration):
            self.sigmas_, self.mus_, self.alphas_ = self.__params()
            labels = self.__labels()
            if labels[labels != self.labels].shape[0] / self.n_samples < self.precision_ratio:
                break
            self.labels = labels
        return self

    def predict(self, x):
        """
        Predict labels for input data in x.

        Args:
            x: Array-like. Data to be predicted. Shape {amount_of_data, dimension}

        Return:
            Array of labels. Shape {amount_of_data,}
        """

        assert x.ndim == 2
        probs = self.__predict(x)
        return np.argmax(probs, axis=0)

    def __initialization(self):
        """
        Initialization all prior parameters.
        """
        alphas = np.full(self.n_clusters, self.dimension)
        means = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        means = self.data[means].copy()
        means = means.reshape([self.n_clusters, self.dimension])
        lambdas = np.ones(self.n_clusters)
        gammas = np.ones(self.n_clusters)
        psis = np.array([np.eye(self.dimension) + 1 for i in range(self.n_clusters)])

        if self.alphas is None: self.alphas = alphas
        if self.means is None: self.means = means
        if self.lambdas is None: self.lambdas = lambdas
        if self.gammas is None: self.gammas = gammas
        if self.psis is None: self.psis = psis

        #self.labels = np.random.choice(self.n_clusters, size=self.n_samples, replace=True)
        km = KMeans.KMeans(self.n_clusters)     # Initialization by K-means.
        self.labels = km.fit_predict(self.data)

    def __gaussian(self, x, mu, sigma):
        """
        Calculate Gaussian prob. of input data x.

        Args:
            x: Array-like data. Shape {amount_of_data, dimension}
            mu: Parameter mean of Gaussian distribution. Shape {1, dimension}
            sigma: Cov-matrix of Gaussian distribution. Shape {dimension. dimension}

        Return:
            Array of log prob., shape {amount_of_data, }
        """

        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        denominator = self.dimension * np.log(2 * np.pi) / 2 + 0.5 * np.log(sigma_det)
        numerator = 0.5 * np.diag(np.dot(np.dot((x - mu), sigma_inv), (x - mu).T))
        log_prob = -numerator - denominator
        return log_prob

    def __predict(self, x):
        """
        Calculate Gaussian prob. of all models of input data x.

        Args:
            x: Array-like data. Shape {amount_of_data, dimension}

        Return:
            Array of log prob., shape {n_cluster, amount_of_data}
        """

        log_prob = []
        for i in range(self.n_clusters):
            mu = self.mus_[i].reshape([1, -1])
            prob = self.__gaussian(x, mu, self.sigmas_[i]) + np.log(self.alphas_[i])
            log_prob.append(prob)
        log_prob = np.array(log_prob)
        return log_prob

    def __params(self):
        """
        Sample sigma, mu from posterior Normal-inverse-Wishart distribution.
        Estimate alpha by posterior mean estimation of Dirichlet distribution. (Collapsed-Gibbs)

        Returns:
            sigmas: sigma in GMM.
            mus: mu in GMM.
            alphas: alpha in GMM.
        """

        values, counts = np.unique(self.labels, return_counts=True)
        alpha = self.alphas.copy()
        alpha[values] += counts[values]
        alpha = alpha / np.sum(alpha)

        mus = []
        sigmas = []
        for i in range(self.n_clusters):
            mask = self.labels == i
            x = self.data[mask]
            ave = np.average(x, axis=0)
            mean = (self.lambdas[i] * self.means[i] + counts[i] * ave) / (self.lambdas[i] + counts[i])
            a = (ave - self.means[i]).reshape([1, self.dimension])
            term1 = np.dot(a.T, a) * (self.lambdas[i] * counts[i]) / (self.lambdas[i] + counts[i])
            b = (x - ave).reshape([counts[i], self.dimension])
            term2 = np.dot(b.T, b)
            psi = self.psis[i] + term1 + term2
            sigma = invwishart.rvs(counts[i] + self.gammas[i], psi)    # Sample from inv-wishart distribution
            sigmas.append(sigma)
            mu = multivariate_normal.rvs(mean, sigma/(counts[i]+self.lambdas[i]))    # Sample from gaussian distribution
            mus.append(mu)
        return np.array(sigmas), np.array(mus), np.array(alpha)

    def __labels(self):
        """
        Sample labels of training data from GMM prob.

        Return:
            Labels of data.
        """

        probs = np.zeros(self.n_samples)
        for i in range(self.n_clusters):
            mu = self.mus_[i].reshape([1, -1])
            prob = self.__gaussian(self.data, mu, self.sigmas_[i]) + np.log(self.alphas_[i])
            probs = np.vstack((probs, prob))
        probs = probs[1:,:]

        labels = []
        for i in range(self.n_samples):
            prob = probs[:, i]
            prob[np.isnan(prob)] = -np.inf
            prob -= np.max(prob)   # scale
            prob = np.exp(prob)
            labels.append(np.random.choice(self.n_clusters, 1, p=list(prob/ np.sum(prob)))[0])

        return np.array(labels)


