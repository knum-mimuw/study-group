import numpy as np
import cvxopt
import itertools
import data_processing


class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, data, labels):
        classes = list(set(labels.T[0]))
        if len(classes) != 2:
            raise ValueError
        y = np.array(list(map(lambda x: 1 if x == classes[0] else -1, labels)))
        no_vectors = data.shape[0]

        P = np.zeros((no_vectors, no_vectors))
        for i, j in itertools.product(range(no_vectors), range(no_vectors)):
            P[i, j] = -np.dot(data[i, :], data[j, :])*y[i]*y[j]
        print(np.linalg.matrix_rank(P), P.shape)
        print(P)
        q = cvxopt.matrix(np.ones((data.shape[0],)))
        P = cvxopt.matrix(P.astype(np.double))
        G = cvxopt.matrix(-np.identity(no_vectors).astype(np.double))
        h = cvxopt.matrix(np.zeros((no_vectors,)).astype(np.double))
        A = cvxopt.matrix(np.diag(y).astype(np.double))
        # A = np.asarray(y)
        b = cvxopt.matrix(h)

        # print(P.shape, q.shape, G.shape, h.shape, A.shape, b.shape)
        alpha = cvxopt.solvers.qp(P, q, G, h, A, b)['x']
        i = max([i if not np.isclose(n, 0) else -1 for i, n in enumerate(alpha)])

        self.w = sum([xi*yi*alphai for xi, yi, alphai in zip(data, y, alpha)], start=np.zeros(no_vectors))
        self.b = 1 - np.dot(sum([a*yi*xi for a, yi, xi in zip(alpha, y, data)], start=np.zeros(data[0].shape)), data[i])

    def predict(self, v):
        return np.dot(self.w, v) + self.b > 0


if __name__ == '__main__':
    data = data_processing.data_generator(no_samples_per_class=2, no_classes=2, no_dimensions=2, centers=((0, 0), (15, 5)))
    svm = SVM()
    svm.fit(*data)
    print(svm.predict(np.array([0, 0])))



