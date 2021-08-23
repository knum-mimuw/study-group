import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle


class NaiveBayes:
    def __init__(self):
        self.tables = dict()

    def fit(self, data, labels):
        classes = set(labels)
        for c in classes:
            data_with_c = filter(lambda t: t[1] == c, zip(data, labels))
            table = np.zeros([2 for _ in range(len(data[0]))])
            for v, _ in data_with_c:
                v = tuple(map(int, v))
                table[v] += 1
            table /= len(labels)
            self.tables[c] = table

    def predict(self, data):
        data = tuple(map(int, data))
        best = (0, None)
        for c in self.tables.keys():
            if self.tables[c][data] > best[0]:
                best = (self.tables[c][data], c)
        return best[1]


if __name__ == '__main__':
    iris = datasets.load_iris()
    vectorized_map = np.vectorize(lambda x, mean: x > mean)
    tmp = iris['data']
    data = np.array([vectorized_map(tmp[:, i], np.mean(tmp[:, i])) for i in range(4)]).T
    labels = iris['target']

    data, labels = list(zip(*shuffle(list(zip(data, labels)))))
    data = np.array(data)
    labels = np.array(labels)
    length = data.shape[0]
    division = .2
    training_data = data[:int(length*division), :]
    test_data = data[int(length*division):, :]
    training_labels = labels[:int(length*division)]
    test_labels = labels[int(length*division):]

    classifier = NaiveBayes()
    classifier.fit(training_data, training_labels)

    good = [classifier.predict(x) == y for x, y in zip(test_data, test_labels)]
    print(f'accuracy: {good.count(True)/len(good)}')
