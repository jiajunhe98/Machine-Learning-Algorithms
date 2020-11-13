import numpy as np
import math
import KMeans

class GaussianMixtureModel:
    """
    This class implements Gaussian Mixture Model(GMM), using EM algorithm to estimate parameters.

    Args:
        x: numpy array contains data, can be 1D or 2D.
        k(int): number of gaussian models(classes).
        max_iteration(int): maximum number of iterations of EM step, default 100.

    Attributes:
        x: numpy array contains data.
        k(int): number of gaussian models(classes).
        dimension: dimension of characteristic space(dimension of data in x).
        alpha: 1D numpy array, contains weight of GMM in log scale.
        mu: 2D numpy array, contains mean of each gaussian model.
        sigma: 3D numpy array, contains covariance of each gaussian model.
        gamma: 2D numpy array, contains the responsibilities of each sample to each model in log scale.

    Methods:
        predict: Predict class labels for input data.

    """
    def __init__(self, x, k, max_iteration = 100):
        """
        Constructor method
        """
        if x.ndim == 1: x = np.reshape(x, [x.shape[0],1])
        self.x = x
        self.k = k
        self.dimension = x.shape[1]
        self.alpha = np.array([- math.log(self.k, math.e) for i in range(self.k)])
        self.mu, self.sigma, self.gamma = self.__initialization()   # alpha, gamma are in log-scale
        iteration = 0
        while iteration < max_iteration:
            mu, sigma, alpha = self.mu, self.sigma, self.alpha
            iteration += 1
            self.__expectation()
            self.__maximization()
            if np.sum((mu - self.mu) ** 2) + np.sum((sigma - self.sigma) ** 2) + np.sum((alpha - self.alpha) ** 2) == 0:
                break

    def predict(self, x):
        """
        Predict class labels for input data.

        Args:
            x: numpy array, contains one or more data.
        Return:
            numpy array of class labels of each data input.
        """
        if x.ndim == 1 and self.dimension == 1: x = np.reshape(x, [x.shape[0], 1]) # characteristic space is 1D, x contains multiple data
        if x.ndim == 1 and self.dimension != 1: x = np.reshape(x, [1, x.shape[0]]) # characteristic space is >1D, x contains 1 data
        if x.ndim == 0 and self.dimension == 1: x = np.reshape(x, [1,1])           # characteristic space is 1D, x contains 1 data
        res = np.zeros(x.shape[0])
        for i in range(self.k):
            prob = self.__gaussian_prob(x, self.mu[i], self.sigma[i]) + self.alpha[i]
            res = np.vstack((res, prob))
        res = res[1:,:]
        return np.argmax(res, axis = 0)

    def __expectation(self):
        """
        E step.
        """
        all_log_prob = np.zeros([self.x.shape[0],1])
        for i in range(self.k):
            prob = self.__gaussian_prob(self.x, self.mu[i], self.sigma[i])
            log_prob = np.reshape(prob, [self.x.shape[0],1]) + self.alpha[i]
            all_log_prob = np.hstack((all_log_prob, log_prob))
        all_log_prob = all_log_prob[:, 1:]
        all_log_prob -= np.reshape(np.max(all_log_prob, axis = 1), [self.x.shape[0], 1])
        all_prob_sum = np.sum(np.exp(all_log_prob), axis = 1)
        all_prob_sum = np.reshape(np.log(all_prob_sum), [self.x.shape[0],1])
        self.gamma = all_log_prob - all_prob_sum

    def __maximization(self):
        """
        M step.
        """
        max_gamma = np.max(self.gamma, axis = 0)
        sum_gamma = np.exp(self.gamma - max_gamma)
        sum_gamma = np.sum(sum_gamma, axis = 0)
        sum_gamma = np.log(sum_gamma) + max_gamma   # denominator for sigma and mu, numerator for alpha, in log scale
        self.alpha = sum_gamma - np.log(self.x.shape[0])

        exp_gamma = np.exp(self.gamma)
        for i in range(self.k):
            numerator_sigma = np.zeros([self.dimension, self.dimension])
            for j in range(self.x.shape[0]):
                diff_x_mu = np.reshape(self.x[j] - self.mu[i], [self.dimension, 1])
                numerator_sigma += np.exp(self.gamma[j][i]) * np.dot(diff_x_mu, diff_x_mu.T)
            numerator_mu = np.sum(self.x * np.reshape(exp_gamma[:,i],[self.x.shape[0],1]), axis = 0)
            self.mu[i] = numerator_mu / np.exp(sum_gamma[i])
            self.sigma[i] = numerator_sigma / np.exp(sum_gamma[i])

    def __gaussian_prob(self, x, mu, sigma):
        """
        Calculate gaussian prob. of input data.

        Args:
             x: 2-dimension numpy array. Each row represents a sample.
             mu: 1-dimension numpy array. Mean of Gaussian distribution.
             sigma: 2-dimension numpy array. Cov matrix of Gaussian distribution.
        Return:
            log prob. of data input in a 1d numpy array.
        """
        if x.ndim == 1: x = np.reshape(x, [1,x.shape[0]])
        sigma_det = np.linalg.det(sigma)
        if sigma_det != 0:
            sigma_inv = np.linalg.inv(sigma)
            log_prob = - self.dimension * np.log(2 * np.pi) / 2 - 0.5 * np.log(sigma_det) - 0.5 * np.diag(np.dot(np.dot((x - mu), sigma_inv), (x-mu).T)) # here assuming x contains multiple data
        else:
            log_prob = np.sum(x - mu, axis = 1)
            log_prob[log_prob != 0] = -math.inf
        return log_prob

    def __initialization(self):
        """
        Initialization all parameters by k-means.

        Returns:
            Initial mu, sigma, gamma
        """
        km = KMeans.KMeans(self.k)
        labels = km.fit_predict(self.x, 50)
        mu = np.array([np.average(self.x[labels == i], axis=0) for i in range(self.k)])
        sigma = np.array([np.eye(self.dimension) + 1 for i in range(self.k)])
        gamma = np.array([[-math.log(self.k, math.e)] * self.k] * self.x.shape[0])
        return mu, sigma, gamma


