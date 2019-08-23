import numpy as np


class PCA:
    def __init__(self, num_bits):
        self.num_bits = num_bits

    def fit(self, X):
        num_rows, num_cols = X.shape

        # Center data.
        self.means = X.mean(axis=0)
        X = self.center(X)

        eigvals, eigvecs = np.linalg.eig(np.dot(X.T, X))

        # Sort the eigenvalues and eigenvectors.
        sort_indices = np.argsort(-eigvals)

        eigvals = eigvals[sort_indices]
        eigvecs = eigvecs[:, sort_indices]

        self.W = eigvecs[:, :self.num_bits]

    def center(self, X):
        return X - self.means

    def transform(self, X):
        return np.dot(self.center(X), self.W)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
