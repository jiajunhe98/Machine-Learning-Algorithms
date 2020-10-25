import numpy as np
import math
import warnings
import matplotlib.pyplot as plt

class OneDimensionalGaussianMixture:
    """
    This class implements one-dimensional gaussian mixture model(1D-GMM), using EM algorithm to calculate the
    parameters of GMM.

    Args:
        x: 1D numpy array, contains data.
        k(int): number of gaussian models(classes).
        max_iteration(int): maximum number of iterations of EM step, default 100.

    Attributes:
        alpha: 1D numpy array, contains alpha parameter (in log scale) of GMM.
        mu: 1D numpy array, contains mean of each gaussian model.
        sigma2: 1D numpy array, contains variance of each gaussian model.
        x: 1D numpy array, contains data.
        k(int): number of gaussian models(classes).
        gamma: 2D numpy array, contains the responsibilities (in log scale) of each sample to each model.

    Methods:
        plot_distribution: Plot input data by histogram and the GMM by probability density diagram.
        get_parameters: Print and return set of parameters of GMM. The parameters contain alpha, sigma and means(mu) for each gaussian model.

    """

    def __init__(self, x, k, max_iteration = 100):
        """
        Constructor method
        """
        self.alpha = np.log(np.array([1/k]*k))   # alpha in log scale
        self.mu = np.array([i * (np.max(x) + np.min(x)) / k for i in range(k)])
        self.sigma2 = np.array([(np.max(x)-np.min(x))**2/10]*k)
        self.x = x
        self.k = k
        self.gamma = self.__E()             # gamma in log scale
        iteration = 0
        while 1:
            indictor = self.__M()
            iteration += 1
            if iteration >= max_iteration or indictor == 1:
                break
            self.gamma = self.__E()

    def plot_distributions(self, scale = False):
        """
        Plot input data by histogram and the GMM by probability density diagram.

        Args:
            scale: If False, the scale of probability density will just follow the real distribution.
                   If True, the probability density (y-axis)  will be scaled to (0,1).

        """
        plt.hist(self.x, density=True, bins = self.x.shape[0], alpha = 0.7)
        x = np.arange(np.min(self.mu - (3 * self.sigma2 ** 0.5)), np.max(self.mu + (3 * self.sigma2 ** 0.5)), 0.0001)
        if scale == False:
            for i in range(self.k):
                if self.sigma2[i] != 0:
                    plt.plot(x, np.exp(self.__gaussian(x,self.mu[i],self.sigma2[i])) * np.exp(self.alpha[i]))
        else:
            for i in range(self.k):
                if self.sigma2[i] != 0:
                    plt.plot(x, np.exp(self.__gaussian(x,self.mu[i],self.sigma2[i]))\
                             / np.exp(self.__gaussian(np.array([self.mu[i]]),self.mu[i],self.sigma2[i])))
        plt.show()

    def get_parameters(self):
        """
        Print and return set of parameters of GMM.
        The parameters contain alpha, sigma and means(mu) for each gaussian model.

        Returns:
             a tuple, contains 3 numpy arrays: array of alpha, sigma and mean for each model.
        """
        print("alpha: ", end = "")
        print(np.exp(self.alpha))
        print("sigma: ", end = "")
        print(self.sigma2**0.5)
        print("means: ", end = "")
        print(self.mu)
        return (np.exp(self.alpha), self.sigma2, self.mu)

    def __E(self):
        """
        E-step.
        Calculate the responsibilities (gamma) of each sample to each model.

        Return:
            A numpy array of gammas.
        """
        res = []
        for i in self.x:
            a = self.__gaussian(i, self.mu, self.sigma2) + self.alpha
            max = np.max(a)
            res.append(np.log(1 / np.sum(np.exp(a - max))) + (a - max)) # in log scale
        return np.array(res)

    def __M(self):
        """
        M-step.
        Re-estimate all parameters.

        Return:
            an indictor showing if parameters are converged.
        """
        return_indictor = 0   # an indictor to show if parameters are converged
        new_sigma2 = []
        new_mu = []
        new_alpha_numerator = []
        for i in range(self.k):
            max_gamma = np.max(self.gamma[:,i])              # find the max term to change log of sum into sum of log
            numerator_sigma = np.dot((self.x - self.mu[i]) ** 2, np.exp(self.gamma[:,i] - max_gamma))
            numerator_mu = np.dot(self.x, np.exp(self.gamma[:,i] - max_gamma))
            new_alpha_numerator.append(max_gamma + np.log(np.sum(np.exp(self.gamma[:,i] - max_gamma))))
            denominator = np.dot(np.array([1]*self.x.shape[0]), np.exp(self.gamma[:,i] - max_gamma))
            new_sigma2.append(numerator_sigma/denominator)
            new_mu.append(numerator_mu/denominator)
        sigma2 = np.array(new_sigma2)
        mu = np.array(new_mu)
        if np.sum((sigma2 - self.sigma2) ** 2) +  np.sum((mu - self.mu) ** 2)  == 0:
            return_indictor = 1
        self.sigma2 = sigma2
        self.mu = mu
        self.alpha =  - np.log(self.x.shape[0]) + new_alpha_numerator
        return return_indictor

    def __gaussian(self, y, mu, sigma2):
        """
        Calculate the log-prob. of getting y from the given gaussian distribution(s).

        Args:
            y: float or numpy array.
            mu: float or numpy array, represents mean(s) of the given distribution(s).
            sigma2: float or numpy array, represents variance(s) of the given distribution(s).


        Return:
            numpy array of log-prob.
        """

        warnings.filterwarnings('ignore')    # ignore warning of nan
        prob = -np.log(2 * np.pi * sigma2) / 2 + (- (y - mu) ** 2 / (sigma2) * 2) # in log-scale
        if mu.ndim != 0:
            """
            Used in EM step, when y is a float and mu and sigma2 are numpy arrays. 
            When plotting, checking is not necessary, for sigma has already been checked to be non-zero.
            """
            prob[np.isnan(prob)] = 0 if y == mu[np.isnan(prob)] else -math.inf
        return np.array(prob)




