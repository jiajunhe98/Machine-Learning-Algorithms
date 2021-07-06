import numpy as np


class FactorAnalysis:
    """
    Gaussian generating model like PPCA but using anisotropy variance.
    """
    def __init__(self, n_components):
        self.M = n_components
        self.W, self.mu, self.Psi = (None,) * 3

    def fit(self, x, max_iter=10000):
        D = x.shape[1]  # Identify the original dimensionality
        self.W = np.random.randn(D, self.M)
        self.mu = np.mean(x, axis=0)
        self.Psi = np.eye(D)

        # EM iteration
        for iter in range(max_iter):
            # E-step
            G = np.linalg.inv(np.eye(self.M) + self.W.T @ np.linalg.inv(self.Psi) @ self.W)
            E_z = G @ self.W.T @ np.linalg.inv(self.Psi) @ (x - self.mu).T

            # M-step
            self.W = ((x - self.mu).T @ E_z.T) @ np.linalg.inv(G * x.shape[0] + E_z @ E_z.T)
            S = np.cov(x)
            self.Psi = np.diag(np.diag(S - self.W @ E_z @ (x - self.mu) / x.shape[0]))

        return self

    def transform(self, x):
        G = np.linalg.inv(np.eye(self.M) + self.W.T @ np.linalg.inv(self.Psi) @ self.W)
        E_z = G @ self.W.T @ np.linalg.inv(self.Psi) @ (x - self.mu).T
        return E_z, G

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)








