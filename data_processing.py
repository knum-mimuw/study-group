import numpy as np
import matplotlib.pyplot as plt
import sklearn


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


if __name__ == '__main__':
    plot_data(data_generator(25, 3, 2, centers=((0, 0), (15, 0), (2, 15))))