import itertools

import numpy as np
import cvxopt

from sklearn import datasets
from sklearn.utils import shuffle

import data_processing


class RetardedSVM:
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


class SVM:
    def __init__(self, softing_parameter=100, kernel=lambda x, y: np.dot(x, y)):
        self.beta = None
        self.C = softing_parameter
        self.kernel = kernel

    def gradient(self, x, y):
        if 1-y*(self.kernel(self.beta, x)) <= 0:
            return self.beta
        return self.beta - self.C*y*x

    def fit(self, data, labels, epochs=10000, learning_rate=lambda t: 1/(1000 + t)):
        # print(labels)
        classes = sorted(list(set(labels)))
        if len(classes) != 2:
            raise ValueError
        pretty_labels = np.array(list(map(lambda x: -1 if x == classes[0] else 1, labels)))
        self.beta = np.random.normal(0, 1, data.shape[1] + 1)

        for i in range(epochs):
            x, y = list(zip(data, pretty_labels))[np.random.randint(0, data.shape[0])]
            x = np.append(x, 1)
            self.beta -= learning_rate(i)*self.gradient(x, y)

    def predict(self, x):
        x = np.append(x, 1)
        return 1 if self.kernel(x, self.beta) >= 0 else -1


if __name__ == '__main__':
    # np.random.seed(1)
    data = data_processing.data_generator(no_samples_per_class=10, no_classes=2, no_dimensions=2, centers=((0, 0), (15, 5)))
    # data_processing.plot_data(data)
    # svm = SVM(softing_parameter=10, kernel=lambda x, y: (np.dot(x, y) + 1000)**10)
    svm = SVM(softing_parameter=10000, kernel=lambda x, y: np.dot(x, y))

    # print(svm.predict(np.array([15, 5])))

    data, target = datasets.load_iris(as_frame=True, return_X_y=True)
    data['target'] = target
    data = data[data['target'] < 2]
    data = shuffle(data)
    print(data.shape)
    training_data = data.iloc[:30, :]
    test_data = data.iloc[30:, :]
    print(len(test_data))

    training_labels = training_data['target'].to_numpy(copy=True)
    test_labels = test_data['target'].to_numpy(copy=True)

    training_data = training_data.drop('target', axis=1)
    test_data = test_data.drop('target', axis=1)

    training_data = training_data.to_numpy(copy=True)
    test_data = test_data.to_numpy(copy=True)

    svm.fit(training_data, training_labels, epochs=20000)

    test_labels = np.where(test_labels == 0, -1, test_labels)
    good = [svm.predict(x) == y for x, y in zip(test_data, test_labels)]
    predictions = [svm.predict(x) for x in test_data]
    print(len(predictions))
    print(f'accuracy: {good.count(True)/len(good)}')




