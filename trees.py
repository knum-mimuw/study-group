import data_processing
import numpy as np
import math
from sklearn import datasets


class Test:
    def __init__(self, labels, target_feature):
        self.labels = labels
        self.target_feature = target_feature
        self.decisions = None
        self.true_child = None
        self.false_child = None

    @staticmethod
    def entropy(values, labels):
        vectors = list(zip(values, labels))

        def ent(o, total):
            z = total - o
            return 0 if z*o == 0 else (-z*math.log2(z/total) - o*math.log2(o/total))/total
        # print(vectors)
        set_sizes = {value: len(list(filter(lambda x: x[0] == value, vectors))) for value in set(values)}
        # print(list(zip(*list(filter(lambda x: x[1] == 0, vectors)))))
        no_ones = {value: sum(list(zip(*list(filter(lambda x: x[0] == value, vectors))))[1]) for value in set(values)}
        return sum([ent(no_ones[value], set_sizes[value]) * set_sizes[value]/len(vectors) for value in set(values)])

    def fit(self, values, labels):
        vectors = list(zip(values, labels))
        set_sizes = {value: len(list(filter(lambda x: x[1] == value, vectors))) for value in set(values)}
        no_ones = {value: sum(list(zip(*filter(lambda x: x[1] == value, vectors)))[1]) for value in set(values)}
        self.decisions = {value: 2*no_ones[value] > set_sizes[value] for value in set(values)}

    def predict(self, vector):
        # print(self.decisions)
        return self.decisions[vector[self.target_feature]]


class IdentificationTree:
    def __init__(self):
        self.root = None

    def fit(self, data, labels):
        target_feature = np.argmax([-Test.entropy(data[:, i], labels) for i in range(data[0, :].shape[0])])
        self.root = Test(labels, target_feature)
        self.root.fit(data[:, target_feature], labels)
        # print(data)
        # print(labels)
        data_labels = np.append(data, np.array([labels]).T, axis=1)

        def dfs(test, data_):
            if test.true_child is not None:
                true_data = np.array(filter(lambda x: x[test.target_feature], data_))
                true_data = np.delete(true_data, test.target_feature, axis=1)
                dfs(test.true_child, true_data)
            if test.false_child is not None:
                not_true_data = filter(lambda x: not x[test.target_feature], data_)
                not_true_data = np.delete(not_true_data, test.target_feature, axis=1)
                dfs(test.false_child, not_true_data)

            true_data = np.array(list(filter(lambda x: x[test.target_feature], data_)))
            if len(set(true_data[:, -1])) <= 1:
                return
            true_data = np.delete(true_data, test.target_feature, axis=1)

            not_true_data = np.array(list(filter(lambda x: not x[test.target_feature], data_)))
            if len(set(not_true_data[:, -1])) <= 1:
                return
            not_true_data = np.delete(not_true_data, test.target_feature, axis=1)
            # print(not_true_data[:, -1])

            t_feature_true = np.argmax([-Test.entropy(true_data[:, i], true_data[:, -1]) for i in range(true_data.shape[1]-1)])
            t_feature_false = np.argmax([-Test.entropy(not_true_data[:, i], not_true_data[:, -1]) for i in range(not_true_data.shape[1]-1)])
            test.true_child = Test(true_data[:, -1], t_feature_true)
            test.false_child = Test(not_true_data[:, -1], t_feature_false)
            test.true_child.fit(true_data[:, t_feature_true], true_data[:, -1])
            test.false_child.fit(not_true_data[:, t_feature_false], not_true_data[:, -1])

            dfs(test.true_child, true_data)
            dfs(test.false_child, not_true_data)

        dfs(self.root, data_labels)

    def predict(self, vector):
        def predict_(test, v):
            decision = test.predict(v)
            if test.true_child is not None:
                if decision:
                    return predict_(test.true_child, v)
                return predict_(test.false_child, v)
            return decision

        return predict_(self.root, vector)


if __name__ == '__main__':
    iris = datasets.load_iris()
    vectorized_map = np.vectorize(lambda x, mean: x > mean)
    tmp = iris['data']
    data = np.array([vectorized_map(tmp[:, i], np.mean(tmp[:, i])) for i in range(4)]).T
    labels = iris['target']
    # print(iris)
    tree = IdentificationTree()
    tree.fit(data[0:99], labels[0:99])
    print(tree.predict(data[100]))



