import numpy as np
import matplotlib.pyplot as plt
import sklearn
import data_processing


class KNN:
    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, v):
        norms = []
        for n, vi in enumerate(self.data):
            norm = sum((vi[i] - v[i])**2 for i in range(vi.shape[0]))
            norms.append([self.labels[n], norm])

        norms = np.array(sorted(norms, key=lambda x: x[1]), dtype=object)
        counts = np.bincount(norms[:self.k, 0].astype('int32'))
        return np.argmax(counts)


if __name__ == '__main__':
    data = data_processing.data_generator(10, 3, 2, centers=((0, 0), (5, 5), (-5, 5)))
    knn_classifier = KNN(k=6)
    knn_classifier.fit(*data)
    print(knn_classifier.predict((0, 0)))

