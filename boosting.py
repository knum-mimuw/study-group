import numpy as np
import math
from sklearn.utils import shuffle
from sklearn import datasets
import trees

from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, classifier, no_classifiers, sampling_ratio, sklearn=False, *args, **kwargs):
        self.classifiers = [classifier(*args, **kwargs) for _ in range(no_classifiers)]
        self.classifiers_weights = []
        self.sampling_ratio = sampling_ratio
        self.possible_labels = None
        self.sklearn = sklearn

    @staticmethod
    def classifiers_error(classifier, data, labels, weights):
        wrong = [i for i, d in enumerate(zip(data, labels)) if classifier.predict([d[0]]) != d[1]]
        return np.sum(weights[wrong])/len(data)

    @staticmethod
    def update_weights(classifier, data, labels, weights):
        wrong = [i for i, d in enumerate(zip(data, labels)) if classifier.predict([d[0]]) != d[1]]
        ok = [i for i, d in enumerate(zip(data, labels)) if classifier.predict([d[0]]) == d[1]]
        weights[wrong] /= (np.sum(weights[wrong])*(2 - int(len(weights[ok]) == 0)))
        weights[ok] /= (np.sum(weights[ok])*(2 - int(len(weights[wrong]) == 0)))

        # print(weights[wrong])
        # print(weights[ok])

    def fit(self, data, labels):
        self.possible_labels = set(labels)
        weight_vector = np.ones(len(data))/len(data)
        for classifier in self.classifiers:
            # print(sum(weight_vector))
            sampled_indices = np.random.choice(np.array(range(len(data))), size=int(self.sampling_ratio*len(data)),
                                               replace=False, p=weight_vector)

            sampled_data = data[sampled_indices]
            sampled_labels = labels[sampled_indices]
            classifier.fit(sampled_data, sampled_labels)
            error = self.classifiers_error(classifier, data, labels, weight_vector)
            self.classifiers_weights.append(math.log((1-error)/error)/2)
            self.update_weights(classifier, data, labels, weight_vector)

    def predict(self, point):
        not_one = list(self.possible_labels - {1})[0]
        v = {
            1: 1,
            not_one: -1
        }
        # print(np.array(self.classifiers_weights))
        return np.sign(np.sum(np.array([v[classifier.predict([point])[0]]
                                        for classifier in self.classifiers]) * np.array(self.classifiers_weights)))

    def change_labels(self, labels, possible_labels=None):
        not_one = list(self.possible_labels - {1})[0] if possible_labels is None else list(possible_labels - {1})[0]
        v = {
            1: 1,
            not_one: -1
        }
        return np.array(list(map(lambda x: v[x], labels)))


if __name__ == '__main__':
    iris = datasets.load_iris()
    vectorized_map = np.vectorize(lambda x, mean: x > mean)
    tmp = iris['data']
    data = np.array([vectorized_map(tmp[:, i], np.mean(tmp[:, i])) for i in range(4)]).T
    labels = iris['target']
    data, labels = list(zip(*shuffle(list(zip(data[:99], labels[:99])))))
    data = np.array(data)
    labels = np.array(labels)

    length = data.shape[0]
    division = .2
    training_data = data[:int(length * division), :]
    test_data = data[int(length * division):, :]
    training_labels = labels[:int(length * division)]
    test_labels = labels[int(length * division):]

    classifier = AdaBoost(DecisionTreeClassifier, 5, .5, sklearn=True)
    classifier2 = DecisionTreeClassifier()
    classifier.fit(training_data, training_labels)
    classifier2.fit(training_data, training_labels)

    good = [classifier2.predict([x]) == y for x, y in zip(test_data, test_labels)]
    print(f'tree accuracy: {good.count(True) / len(good)}')

    test_labels = classifier.change_labels(test_labels)

    good = [classifier.predict(x) == y for x, y in zip(test_data, test_labels)]
    print(f'boosting accuracy: {good.count(True) / len(good)}')


