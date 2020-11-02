import numpy as np

def distances(x, distance = "E", p = None):
    """
    Take a data matrix, return the distances matrix.

    Args:
        x: numpy array, data matrix.
        distance: {"E": Euclidean Distance, "Minkowski": Minkowski Distance, "Manhattan": Manhattan Distance,
                "C": Chebyshev Distance, "Mahalanobis": Mahalanobis Distance, "Corr": Correlation Coefficient,
                "cos": cosine}, default "E".
        p: If Minkowski distance is chosen, p should be set, default None.

    Return:
        Distances matrix.
    """
    if x.ndim == 1: x = np.reshape(x, [1, x.shape[0]])
    if x.ndim == 0: x = np.reshape(x, [1, 1])

    x = x.astype(np.float64)

    if distance == "Minkowski" and p == None:
        raise TypeError("Missing p for Minkowski distance.")

    if distance in  {"E", "Minkowski", "Manhattan", "C"}:
        if distance == "Manhattan": p = 1
        if distance == "E": p = 2
        res = np.zeros([1, x.shape[0]])
        for i in x:
            dis = np.sum(np.abs(x - i) ** p, axis = 1) ** (1/p) if distance != "C" else np.max(np.abs(x - i), axis = 1)
            dis = np.reshape(dis, [1, x.shape[0]])
            res = np.vstack([res, dis])
        return res[1:]

    if distance == "Mahalanobis":
        cov = np.cov(x.T)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            raise ValueError("Cov matrix is singular matrix")
        res = np.zeros([1, x.shape[0]])
        for i in x:
            diff = x - i
            dis = np.diag(np.dot(np.dot(diff, inv), diff.T))
            res = np.vstack([res, dis])
        return res[1:]

    if distance == "Corr":
        return np.corrcoef(x)

    if distance == "cos":
        res = np.dot(x, x.T)
        for i in range(x.shape[0]):
            factor = res[i,i] ** 0.5
            res[:,i] = res[:,i] / factor
            res[i,:] = res[i,:] / factor
        return res



















