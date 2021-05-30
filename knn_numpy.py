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


def plot_knn_results(data, k):
    knn_classifier = KNN(k=k)
    knn_classifier.fit(*data)
    sup_x = (np.min(data[0][:, 0]), np.max(data[0][:, 0]))
    sup_y = (np.min(data[0][:, 1]), np.max(data[0][:, 1]))
    results = np.array([[knn_classifier.predict((x, y)) for y in range(*sup_y)] for x in range(*sup_x)])
    points, labels = data
    colors_dict = {0: 'red', 1: 'blue', 2: 'green'}
    brush = np.vectorize(lambda x: colors_dict[x])
    colors = brush(labels)
    background = brush(results)
    plt.pcolormesh(background)
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()


if __name__ == '__main__':
    data = data_processing.data_generator(10, 3, 2, centers=((0, 0), (5, 5), (-5, 5)))
    plot_knn_results(data, 7)

