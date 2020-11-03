import numpy as np

class PCA:
    """
    This class implements PCA.

    Args:
        k(int): Number of principle components. If None, k will be set equal to the original dimension.
        whiten(boolean): When True (False by default) the components are multiplied by the square root of (numbers of
                         data - 1) and then divided by the singular values to ensure uncorrelated outputs with
                        unit variances.

    Attributes:
        k(int): Number of principle components. If None, k will be set equal to the original dimension.
        whiten(boolean): whiten or not.
        x: Array of data.
        mean: Array of mean of data.
        components: Array of principle components.
        singular_values: Array of singular values of input data array. Variances after PCA is equals to
                        singular_values ** 2 / (numbers of data - 1).

    Methods:
        PCA.fit: Fit the model with data.
        PCA.transform: Apply PCA model to data.
        PCA.fit_transform: Fit and apply.
        PCA.inverse_transform: Transform data back to its original space.
        PCA.factor_loading: Return the factor loading for all principle components to all dimension.
        PCA.explained_variance: Return the amount/ratio of variance explained by each of the selected components.
    """

    def __init__(self, k = None, whiten = False):
        """
        Constructor method.
        """

        self.k = k
        self.whiten = whiten
        self.x = None
        self.mean = None
        self.components = None
        self.singular_values = None

    def fit(self, x):
        """
        Fit the model with data.

        Args:
            x: array of data.
        """

        self.x = x.copy()
        self.x = self.__check_x(self.x)
        self.mean = np.average(self.x, axis = 0)
        self.x -= self.mean
        self.k = self.x.shape[1] if self.k == None else self.k
        U, S, V = np.linalg.svd(self.x)
        self.singular_values = S[:self.k]
        self.components = V[:self.k, :]
        self.__original_singular_values = S
        self.__original_components = V


    def transform(self, x):

        """
        Apply PCA model to data.

        Args:
            x: array of data.

        Return:
            array of data represented bt the principal components.
        """

        x = self.__check_x(x) - self.mean
        res = np.dot(x, self.components.T) if self.whiten == False \
            else (self.x.shape[0]-1)**0.5 * np.dot(x, self.components.T) / self.singular_values
        return res

    def fit_transform(self, x):
        """
        Fit and apply.

        Args:
            x: array of data.

        Return:
            array of data represented bt the principal components.
        """

        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):

        """
        Transform data back to its original space.

        Args:
            x: array of data.

        Returns:
            data in original space.
        """
        x *= self.singular_values / (self.x.shape[0] - 1) ** 0.5 if self.whiten else 1
        return np.dot(x, self.components) + self.mean


    def factor_loading(self):
        """
        Return the factor loading for all principle components to all dimension.

        Return:
            Array-like, shape(number_of_principle_components, number_of_original_dimensions)
        """

        fl = self.components * np.reshape(self.singular_values, [self.k,1])
        fl /= np.diag(np.dot(self.x.T, self.x)) ** 0.5
        return fl

    def explained_variance(self, ratio = True):
        """
        Return the amount/ratio of variance explained by each of the selected components.

        Args:
            ratio: If True, return ratio of variance. If False, returnthe value of variance. Default True.

        Return:
            Array of variance (value/ratio).
        """

        var = self.singular_values ** 2
        all_var = self.__original_singular_values ** 2
        return var / (self.x.shape[0]-1) if not ratio else var / np.sum(all_var)

    def __check_x(self, x):

        """
        Check if x is 2D array. If not, convert.
        Check if dimension matches. If not, raise error.

        Return:
            x in 2D array.
        """

        if x.ndim == 1: x = np.reshape(x, [1, x.shape[0]])
        if x.ndim == 0: x = np.reshape(x, [1, 1])
        if self.k != None and x.shape[1] < self.k: raise ValueError("Dimension should not be smaller than k.")
        if self.x.shape[1] != x.shape[1]: raise  ValueError("Dimension not match.")
        return x
