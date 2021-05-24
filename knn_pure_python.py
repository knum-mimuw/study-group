import matplotlib.pyplot as plt
import numpy as np
import data_processing
from sklearn import datasets


class KNN:
    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = list(map(lambda x: x[0], labels))

    def predict(self, v):
        # norms = []
        # for n, vi in enumerate(self.data):
        #     norm = sum((vi[i] - v[i])**2 for i in range(len(vi)))
        #     norms.append([self.labels[n], norm])

        norms = [[self.labels[n], sum((vi[i] - v[i])**2 for i in range(len(vi)))] for n, vi in enumerate(self.data)]
        norms = sorted(norms, key=lambda x: x[1])[:self.k]
        # labels = list(map(lambda x: x[0], norms))
        labels = list(zip(*norms))[0]
        return max(set(labels), key=labels.count)

        # return max(set(map(lambda x: x[0], sorted([[self.labels[n], sum((vi[i] - v[i])**2 for i in range(len(vi)))] for n, vi in enumerate(self.data)], key=lambda x: x[1])[:self.k])), key=list(map(lambda x: x[0], sorted([[self.labels[n], sum((vi[i] - v[i])**2 for i in range(len(vi)))] for n, vi in enumerate(self.data)], key=lambda x: x[1])[:self.k])).count)


if __name__ == '__main__':
    data = data_processing.data_generator(10, 3, 2, centers=((0, 0), (5, 5), (-5, 5)))
    data = list(map(lambda x: x.tolist(), data))
    knn_classifier = KNN(k=6)
    knn_classifier.fit(*data)
    print(knn_classifier.predict((0, 0)))
