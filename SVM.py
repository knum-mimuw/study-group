import numpy as np
import cvxopt
import itertools

class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, data, labels):
        classes = list(set(labels))
        if len(classes) != 2:
            raise ValueError
        y = np.array(list(map(lambda x: 1 if x == classes[0] else -1, labels)))
        no_vectors = data.shape[0]

        P = np.zeros((no_vectors, no_vectors))
        for i, j in itertools.product(range(no_vectors), range(no_vectors)):
            P[i, j] = -np.dot(data[i, :], data[j, :])*y[i]*y[j]

        q = np.ones((data.shape[1],))
        P = P.astype('float32')
        G = -np.identity(no_vectors).astype('float32')
        h = np.zeros((no_vectors,)).astype('float32')
        A = y.astype('float32')
        # A = np.asarray(y)
        b = h

        alpha = cvxopt.solvers.qp(P, q, G, h, A, b)['x']

        self.w = sum([xi*yi*alphai for xi, yi, alphai in zip(data, y, alpha)], start=np.zeros(no_vectors))

    def predict(self, v):
        pass

