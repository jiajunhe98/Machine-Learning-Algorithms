import numpy as np


class BinaryClassificationSVM(object):
    """
    This class implements binary classification SVM.

    Args:
        C(float): Regularization factor. Default 0.
        kernel(str): Kernel function. {"linear": linear kernel, "poly": polynomial kernel,
                                        "rbf": radial basis kernel, "sigmoid": sigmoid kernel.}
        gamma: Parameter in poly/rbf/sigmoid kernel.
        degree: Parameter in poly kernel.
        coef: Constant parameter in poly/sigmoid kernel.

    Attributes:
        alpha: coefficient of each vector in classifier.
        b: constant coefficient in classifier.

    Methods:
        fit(x, y, max_iteration=100): Fit the SVM model according to the given training data.
        predict(x): Perform classification on samples in X.


    """

    def __init__(self, C=0.0, kernel="linear", gamma=None, degree=3, coef=1):
        """
        Constructor method.
        """

        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef = coef
        self.alpha = None
        self.b = 0
        self.x = None
        self.y = None
        self.kernel_matrix = None

    def fit(self, x, y, max_iteration=100):
        """
        Fit the SVM model according to the given training data.

        Args:
            x: Array of data. Shape {amount_of_data, number_of_dimension_of_each_data}
            y: Labels of data. Class should be represented by -1/1. Shape {amount_of_data, }
            max_iteration(int): Maximum iteration times of SMO algorithms. Default 100.

        Return:
            object: SVM model
        """

        self.x = x
        self.y = y
        self.alpha = np.zeros([y.shape[0]])
        self.kernel_matrix = self.__kernel(self.x, self.x)
        for i in range(max_iteration):
            if self.__iteration() == -1:
                break

        return self

    def predict(self, x):
        """
        Perform classification on samples in X.

        Return:
            Labels of data. Shape {amount_of_data, }

        """
        kernel = self.__kernel(x, self.x)
        gx = np.sum(kernel * self.alpha * self.y, axis=1) + self.b

        return np.sign(gx).astype(np.int)

    def __iteration(self):
        """
        Find alpha1 alpha2 and update the value.
        """
        new_index_1 = None
        gx = np.sum(self.kernel_matrix * self.alpha * self.y, axis=1) + self.b
        ex = gx - self.y
        for i in range(self.alpha.shape[0]):
            if 0 < self.alpha[i] < self.C and gx[i] * self.y[i] != 1:
                new_index_1 = i
                break
        if new_index_1 is None:
            for i in range(self.alpha.shape[0]):
                if self.alpha[i] == 0 and gx[i] * self.y[i] < 1:
                    new_index_1 = i
                    break
                if self.alpha[i] == self.C and gx[i] * self.y[i] > 1:
                    new_index_1 = i
                    break
        if new_index_1 is None:
            return -1
        _ex = np.abs(ex.copy() - ex[new_index_1])
        _ex[new_index_1] = -np.inf
        new_index_2 = np.argmax(_ex)
        eta = self.kernel_matrix[new_index_1, new_index_1] + \
              self.kernel_matrix[new_index_2, new_index_2] - \
              self.kernel_matrix[new_index_1, new_index_2] * 2
        new_value_2 = self.y[new_index_2] * (ex[new_index_1] - ex[new_index_2]) / eta + self.alpha[new_index_2]
        if self.y[new_index_1] != self.y[new_index_2]:
            _l = max(0, self.alpha[new_index_2] - self.alpha[new_index_1])
            _h = min(self.C, self.C + self.alpha[new_index_2] - self.alpha[new_index_1])
        else:
            _l = max(0, self.alpha[new_index_2] + self.alpha[new_index_1] - self.C)
            _h = min(self.C, self.alpha[new_index_2] + self.alpha[new_index_1])
        if new_value_2 > _h:
            new_value_2 = _h
        if new_value_2 < _l:
            new_value_2 = _l
        new_value_1 = self.y[new_index_1] * self.y[new_index_2] * (self.alpha[new_index_2] - new_value_2) + \
                      self.alpha[new_index_1]
        b1 = -ex[new_index_1] - \
             self.y[new_index_1] * self.kernel_matrix[new_index_1, new_index_1] * (new_value_1 - self.alpha[new_index_1]) - \
             self.y[new_index_2] * self.kernel_matrix[new_index_2, new_index_1] * (new_value_2 - self.alpha[new_index_2]) + \
             self.b
        b2 = -ex[new_index_2] - \
             self.y[new_index_1] * self.kernel_matrix[new_index_1, new_index_2] * (new_value_1 - self.alpha[new_index_1]) - \
             self.y[new_index_2] * self.kernel_matrix[new_index_2, new_index_2] * (new_value_2 - self.alpha[new_index_2]) + \
             self.b
        self.b = (b1 + b2) / 2
        self.alpha[new_index_1] = new_value_1
        self.alpha[new_index_2] = new_value_2

        return 0

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
            self.gamma = 1 / self.x.shape[1]
        if self.kernel == "poly":
            res = np.dot(x, y.T) * self.gamma + self.coef
            res = res ** self.degree
        if self.kernel == "rbf":
            res = np.array([np.sum((y - x[i]) ** 2, axis=1) for i in range(x.shape[0])])
            res = np.exp(-self.gamma * res[1:])
        if self.kernel == "sigmoid":
            res = np.dot(x, y.T) * self.gamma - self.coef
            res = np.tanh(res)
        return np.reshape(res, [x.shape[0], y.shape[0]])  # in case that x/y contains only one data.


svm = BinaryClassificationSVM(C=0.5, kernel="poly")
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split

mask = np.any(np.vstack((skd.load_digits().target == 0, skd.load_digits().target == 5)), axis=0)

iris = skd.load_digits().target[mask]
iris[iris == 0] = -1
iris[iris == 5] = 1
data = skd.load_digits().data[mask]

X_train, X_test, y_train, y_test = train_test_split(
    data, iris, test_size=0.4, shuffle=True)

svm.fit(X_train, y_train)
print(svm.predict(X_test) == y_test)


