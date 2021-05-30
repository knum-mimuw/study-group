import data_processing
import numpy as np
import math
from sklearn import datasets


class Test:
    def __init__(self, label, label_index):
        self.label = label
        self.target_feature = label_index
        self.decisions = None
        self.true_child = None
        self.false_child = None

    @staticmethod
    def entropy(values, labels):
        vectors = list(zip(values, labels))

        def ent(o, total):
            z = total - o
            return 0 if z*o == 0 else (-z*math.log2(z/total) - o*math.log2(o/total))/total

        set_sizes = {value: len(list(filter(lambda x: x[1] == value, vectors))) for value in set(values)}
        no_ones = {value: sum(list(zip(*filter(lambda x: x[1] == value, vectors)))[1]) for value in set(values)}
        return sum([ent(no_ones[value], set_sizes[value]) * set_sizes[value]/len(vectors) for value in set(values)])

    def fit(self, values, labels):
        vectors = list(zip(values, labels))
        set_sizes = {value: len(list(filter(lambda x: x[1] == value, vectors))) for value in set(values)}
        no_ones = {value: sum(list(zip(*filter(lambda x: x[1] == value, vectors)))[1]) for value in set(values)}
        self.decisions = {value: 2*no_ones[value] > set_sizes[value] for value in set(values)}

    def predict(self, vector):
        return self.decisions[vector[self.target_feature]]


class IdentificationTree:
    def __init__(self):
        self.root = None

    def fit(self, data, labels):
        target_feature = np.argmax([-Test.entropy(data[:, i], labels) for i in range(data[0, :].shape[0])])
        self.root = Test(labels, target_feature)
        data_labels = np.concatenate(data, labels, axis=1)

        def dfs(test, data_):
            if len(set(data_[:, -1])) == 1:
                return
            if test.true_child is not None:
                true_data = np.array(filter(lambda x: x[test.target_feature], data_))
                true_data = np.delete(true_data, test.target_feature, axis=1)
                dfs(test.true_child, true_data)
            if test.false_child is not None:
                not_true_data = filter(lambda x: not x[test.target_feature], data_)
                not_true_data = np.delete(not_true_data, test.target_feature, axis=1)
                dfs(test.false_child, not_true_data)


if __name__ == '__main__':
    iris = datasets.load_iris()
    vectorized_map = np.vectorize(lambda x, mean: x > mean)
    tmp = iris['data']
    data = np.array([vectorized_map(tmp[:, i], np.mean(tmp[:, i])) for i in range(4)]).T
    labels = iris['target']
    tree = IdentificationTree()
    tree.fit(data, labels)
