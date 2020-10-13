import numpy as np
class Perceptron(object):
    """
    This class implement Perceptron for binary classification.

    Attributes:
        w: a numpy array of weight
        b(float): bias

    Methods:
        train(x, y, learning_rate = 1, max_iteration = 1000, max_misclassified = 0): Train Perceptron by gradient descending.
        predict(x)
    """
    def __init__(self):
        """
        Constructor method
        """
        self.w = None
        self.b = 0

    def train(self, x, y, learning_rate = 1, max_iteration = 1000, max_misclassified = 0):
        """
        Train Perceptron by gradient descending.

        Args:
            x: 2-dimension numpy array, characters of each sample, each row represents a sample
            y: 1-dimension numpy array, class label of each sample, y belongs to {-1, +1}
            learning_rate(float,optional): learning rate between (0,1], default 1
            max_iteration(int,optional): maximum times of iteration, default 1000
            max_misclassified(int,optional): maximum number of mis-classified samples which can be tolerated, default 0

        Raises:
            'ValueError': x[:,0] and y should have same dimension

        Returns:
            None
        """
        if y.shape[0] != x.shape[0]:
            raise ValueError("x[:,0] and y should have same dimension")
        sample_num = y.shape[0]
        dimension = x.shape[1]
        iteration = 0
        self.w = np.array([0] * dimension)
        while iteration <= max_iteration:
            misclassified_num = 0
            for i in range(sample_num):
                iteration += 1
                if y[i]* (np.dot(self.w, x[i])+self.b) <= 0:
                    self.w += learning_rate * y[i] * x[i]
                    self.b += learning_rate * y[i]
                    misclassified_num += 1
            if misclassified_num <= max_misclassified:
                break

    def predict(self,x):
        """
        Predict class for a brunch of samples.

        Args:
            x: 2-dimension numpy array, samples to be predicted, each row represent a sample

        Raises:
            'ValueError': Perceptron have not been trained yet
            'ValueError': samples in x should have same dimension as the characteristic space


        Returns:
            a numpy array of the class of each sample input
        """
        if self.w == None:
            raise ValueError("Perceptron have not been trained yet")
        if self.w.shape[0] != x.shape[1]:
            raise ValueError("dimension of characteristic space does not match")
        res = np.array([-1]*x.shape[0])
        res[np.dot(self.w, x.T) + self.b >=0] = 1
        return res


