import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.utils import shuffle
from sklearn import datasets

import boosting
from sklearn.tree import DecisionTreeClassifier


def data_generator(no_samples_per_class, no_classes, no_dimensions=2, centers=None, std_list=None, random_std=False,
                   std=3, space_size=10):
    if centers is None:
        centers = np.random.random((no_classes, no_dimensions))*2*space_size - space_size
    if std_list is None and random_std:
        std = np.random.normal(loc=std if std is not None else 1, size=(no_classes, no_dimensions))
    std = std_list if std is None else std

    return np.concatenate([np.random.normal(centers[i], std, (no_samples_per_class, no_dimensions))
                           for i in range(no_classes)]), \
           np.concatenate([np.full((no_samples_per_class, 1), i) for i in range(no_classes)])


def plot_data(data):
    points, labels = data
    colors_dict = {0: 'red', 1: 'blue', 2: 'green'}
    colors = tuple(map(lambda label: colors_dict[label[0]], labels))
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()


class CrossValidation:
    def __init__(self, k, classifier):
        self.k = k
        self.classifiers = [copy.deepcopy(classifier) for _ in range(k)]

    def evaluate(self, data, labels):
        parts = []
        accuracies = []
        for i in np.linspace(0, len(data)//self.k*self.k, self.k, endpoint=False):
            i = int(i)
            parts.append(list(zip(data, labels))[i:i + len(data)//self.k])
        for i in range(self.k):
            # print(parts[:i]+parts[i+1:])
            d, lb = list(map(np.array, zip(*(sum(parts[:i]+parts[i+1:], start=[])))))
            self.classifiers[i].fit(d, lb)
            # print(parts[i])
            acc = np.mean([int(self.classifiers[i].predict(data_point) == label) for data_point, label in parts[i]])
            accuracies.append(acc)
        return np.mean(accuracies)


if __name__ == '__main__':
    iris = datasets.load_iris()
    vectorized_map = np.vectorize(lambda x, mean: x > mean)
    tmp = iris['data']
    data = np.array([vectorized_map(tmp[:, i], np.mean(tmp[:, i])) for i in range(4)]).T
    labels = iris['target']
    data, labels = list(zip(*shuffle(list(zip(data[:99], labels[:99])))))
    data = np.array(data)
    labels = np.array(labels)

    classifier = boosting.AdaBoost(DecisionTreeClassifier, 5, .5, sklearn=True)
    labels = classifier.change_labels(labels, possible_labels={1, 0, -1})

    validator = CrossValidation(5, classifier)
    print(validator.evaluate(data, labels))

