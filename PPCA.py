import numpy as np


class PPCA:

    def __init__(self):
        self.dim = 0
        self.latent_dim = 0
        self.W = None
        self.mu = None
        self.sigma2 = None
        self.alpha = None

    def fit(self, data, alpha_threshold=1e3, alpha_max_iter=10, alpha_iter_threshold=1e-6, EM_max_iter=200, EM_threshold=1e-6, random_state=None):

        np.random.seed(random_state)
        self.dim = data.shape[1]
        self.latent_dim = self.dim - 1
        self.W = np.random.randn(self.dim, self.latent_dim)
        self.mu = np.mean(data, axis=0)
        self.sigma2 = np.random.rand(1)
        self.alpha = np.random.rand(self.latent_dim)

        for iter_alpha in range(alpha_max_iter):

            for iter_EM in range(EM_max_iter):

                old_W = self.W

                # E-step
                M = self.W.T @ self.W + self.sigma2 * np.eye(self.latent_dim)
                E_Z = np.linalg.inv(M) @ self.W.T @ (data - self.mu).T
                E_ZZ = data.shape[0] * self.sigma2 * np.linalg.inv(M) + E_Z @ E_Z.T

                # M-step
                self.W = (data - self.mu).T @ E_Z.T @ np.linalg.inv(np.diag(self.alpha) * self.sigma2 + E_ZZ)
                self.sigma2 = 1 / self.dim / data.shape[0] * (np.sum((data - self.mu) ** 2) -
                                                          2 * np.trace(E_Z.T @ self.W.T @ (data - self.mu).T) +
                                                          np.trace(E_ZZ @ self.W.T @ self.W))

                # check the threshold
                if np.sqrt(np.sum((self.W - old_W) ** 2)) <= EM_threshold:
                    break

            # Calculate alpha
            old_alpha = self.alpha
            self.alpha = self.dim / (np.diag(self.W.T @ self.W) + 1e-10)
            self.latent_dim = np.sum(self.alpha <= alpha_threshold)
            self.W = self.W[:, self.alpha <= alpha_threshold]
            self.alpha = self.alpha[self.alpha <= alpha_threshold]
            if self.alpha.shape[0] == old_alpha.shape[0] and np.linalg.norm(self.alpha - old_alpha) <= alpha_iter_threshold:
                break

        return self

    def transform(self, data):

        M = self.W.T @ self.W + self.sigma2 * np.eye(self.latent_dim)
        Z = (data - self.mu) @ self.W @ np.linalg.inv(M).T

        return Z


z = np.random.randn(1500, 5)
W = np.random.randn(100, 5) * 20
mu = np.random.rand(100) * 10
data = z @ W.T + mu + np.random.randn(100)

ppca = PPCA().fit(data, random_state=42)
print(ppca.latent_dim)
print(ppca.alpha)