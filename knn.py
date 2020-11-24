import numpy as np


def knn(x, dataset, labels, k):
    """
    Input dataset and find the label of x.
    All calculation is using array broadcast without any explicit loop.

    Args:
        x: Array-like. Data to be predicted. Shape (amount_of_data, dim_of data).
        dataset: Array-like. Training dataset. Shape (amount_of_training_data, dim_of_data).
        labels: Array-like. Labels of training data. Shape (amount_of_training_data,)
        k(int): How many adjacent data to be considered when predicting the labels of x.

    Returns:
        Labels of data, shape (amount_of_data,)
    """
    assert x.ndim == 2
    amount_of_training_data = dataset.shape[0]
    x_all = np.vstack((x.T,) * amount_of_training_data).reshape((amount_of_training_data, x.shape[1], x.shape[0]))
    x_all = (x_all.T - dataset.T) ** 2
    x_all = np.sum(x_all, axis=1).T
    x_indices = np.argsort(x_all, axis=0)
    x_labels = np.vstack((labels,) * x.shape[0]).T
    x_labels = np.take_along_axis(x_labels, x_indices, axis=0)[:k, :]
    res = np.apply_along_axis(lambda y: np.argmax(np.bincount(y.astype(int))), axis=0, arr=x_labels)
    return res
