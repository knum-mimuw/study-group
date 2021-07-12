import numpy as np
from scipy import linalg

REL_TOL = 1e-5


class SVD:
    def __init__(self, matrix):
        eig, self.V = np.linalg.eig(matrix.T @ matrix)
        sigma_kwadrat, self.V = list(zip(*sorted(list(zip(eig, self.V.T)), key=lambda x: x[0], reverse=True)))
        self.V = np.asarray(self.V)
        self.V = np.real(linalg.orth(self.V))
        sigma_kwadrat = list(map(lambda x: 0 if abs(x) < REL_TOL else x, sigma_kwadrat))
        self.sigma = np.sqrt(sigma_kwadrat)
        U = matrix @ self.V
        self.U = np.array([u/sigma for u, sigma in zip(U.T, self.sigma) if sigma != 0]).T

    def get_matrix(self):
        sigma = np.zeros((self.U.shape[0], self.V.shape[0]))
        sigma_diag = np.diag(np.array(list(filter(lambda x: x != 0, self.sigma))))
        sigma[:sigma_diag.shape[0], :sigma_diag.shape[0]] = sigma_diag
        return self.U @ sigma @ self.V.T


if __name__ == '__main__':
    m = np.array([[1, 3, 1, 1], [0, 1, 2, 3], [2, 23, 2, 3], [2, 1, 2, 10]])
    svd = SVD(m)
    print(svd.get_matrix().round().astype('int8'))
    print()
    print(m)

    # print(np.linalg.svd(m))
