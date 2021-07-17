import numpy as np
import math

from animator import basic_func, objects


def epoch_frame(target_func, approximators_list, generate_png=False):
    """
    Plotting family of functions belonging to single epoch and the target function.

    Args:
        target_func (function): Function to be approximated.
        approximators_list (list of functions): Approximators from one epoch.
        generate_png (bool, optional): Saves the image to 'frame.png' if True.

    Returns:
        basic_func.OneAxisFrame: Generated frame.
    """
    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    target_function = objects.Function(target_func)
    approximators = [objects.Function(foo) for foo in approximators_list]

    settings_target_function = {
        'sampling rate': 3,
        'thickness': 8,
        'blur': 3,
        'color': 'white'
    }
    settings_approximators = {
        'sampling rate': 3,
        'thickness': 4,
        'blur': 2,
        'color': 'light gray'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 2,
        'blur': 1,
        'color': 'white'
    }

    frame.add_axis_surface(x_bounds=(0, 1), y_bounds=(-2, 2))
    frame.add_axes(settings_axes, x_only=True)
    for foo in approximators:
        frame.blit_parametric_object(foo, settings_approximators)
    frame.blit_parametric_object(target_function, settings_target_function)
    frame.blit_axis_surface()
    if generate_png:
        frame.generate_png(f'frame.png')
    return frame


def make_film(target_func, epochs, filename='genetic.mp4', fps=1, resolution=(1280, 720), step=1):
    """
    Generates the video illustrating function approximation by genetic algorithm.

    Args:
        target_func (function): Function to be approximated.
        epochs (list): List of epochs, where epoch is a list of approximators belonging to that epoch.
        filename (str): Target filename.
        fps (int): Frames per second.
        resolution: Resolution of the video. Should be the same as frame resolution.
        step (int): Skip every step epochs in the video.
    """
    video = basic_func.Film(fps, resolution)
    for i, epoch in enumerate(epochs[::step]):
        video.add_frame(epoch_frame(target_func, epoch))
        print(i)
    video.render(filename)


def make_polynomial(coef):
    return lambda x: sum(a*x**i for i, a in enumerate(coef))


def make_random_generation(n=10):
    return [make_polynomial([np.random.normal() for _ in range(10)]) for _ in range(n)]


def make_random_epochs(n=10):
    return [make_random_generation() for _ in range(n)]


class Approximant:
    def __init__(self, length, values=None, std=None):
        self.length = length
        self.values = np.random.normal(0, 1, (length,))
        self.std = np.abs(np.random.normal(0, 1, (length,)))
        if values is not None:
            self.values = values
            self.std = std
        self.temp_fitness_rank = None
        self.temp_diversity_rank = None

    def make_polynomial(self):
        return lambda x: sum(a*x**i for i, a in enumerate(self.values))

    def sex(self, other):
        t = np.random.random()
        return Approximant(self.length, values=t*self.values + (1-t)*other.values,
                           std=t*self.std + (1-t)*other.std)

    def mutate(self):
        self.values += np.random.normal(0, self.std, (self.length,))
        self.std += np.random.normal(np.zeros((self.length,)), self.std, (self.length,))
        self.std = np.abs(self.std)
        return self

    def fitness(self, target_foo, norm='sup', sampling_rate=100):
        if norm == 'sup':
            return 1/(max([abs(target_foo(x)-self.make_polynomial()(x)) for x in np.linspace(0, 1, sampling_rate)])+1)
        if norm == 'int':
            return 1/(sum([abs(target_foo(x) - self.make_polynomial()(x)) for x in np.linspace(0, 1, sampling_rate)])+1)

    def diversity(self, new_population):
        return np.mean([np.linalg.norm(self.values, x.values) for x in new_population])


class Population:
    def __init__(self, size, unit_length, target_function, society=None):
        self.size = size
        self.unit_length = unit_length
        self.society = society if society is not None else [Approximant(unit_length) for _ in range(size)]
        self.target_function = target_function

    def group_sex(self, selection_type='proportional', p_c=.6):
        self.society.sort(key=lambda x: x.fitness(self.target_function), reverse=True)

        for i, x in enumerate(self.society):
            x.temp_fitness_rank = i

        if selection_type == 'diversity rank':
            distribution = [(1 - p_c) ** k * p_c for k in range(self.size - 1)] + [(1 - p_c) ** (self.size - 1)]
            parents = np.random.choice(self.society, 2, replace=True, p=distribution)
            new_population = [parents[0].sex(parents[1]).mutate()]

            while len(new_population) != self.size:
                self.society.sort(key=lambda guy: guy.diversity(new_population), reverse=True)
                for i, x in enumerate(self.society):
                    x.temp_diversity_rank = i
                self.society.sort(key=lambda guy: guy.temp_fitness_rank**2 + guy.temp_diversity_rank**2)
                parents = np.random.choice(self.society, 2, replace=True, p=distribution)
                new_population += [parents[0].sex(parents[1]).mutate()]
            return Population(self.size, self.unit_length, self.target_function, new_population)

        if selection_type == 'proportional':
            sum_of_fitness = sum([x.fitness(self.target_function) for x in self.society])
            distribution = [x.fitness(self.target_function)/sum_of_fitness for x in self.society]

        else:
            distribution = [(1-p_c)**k*p_c for k in range(self.size-1)] + [(1-p_c)**(self.size-1)]

        new_mothers = np.random.choice(self.society, 2 * self.size, replace=True, p=distribution)
        new_fathers = np.random.choice(self.society, 2 * self.size, replace=True, p=distribution)
        new_population = [x.sex(y).mutate() for x, y in zip(new_fathers, new_mothers)]
        return Population(self.size, self.unit_length, self.target_function, new_population)

    def census(self):
        return [make_polynomial(x.values) for x in self.society]

    def get_best_fitness(self):
        self.society.sort(key=lambda guy: guy.fitness(self.target_function), reverse=True)
        return self.society[0].fitness(self.target_function)


def genetic_algorithm(target_function, population_size, unit_length, epochs):
    populations = [Population(population_size, unit_length, target_function)]
    for _ in range(epochs):
        populations.append(populations[-1].group_sex())
        print(populations[-1].get_best_fitness())
    return populations


if __name__ == '__main__':
    populations = genetic_algorithm(lambda x: math.sin(10*x), 20, 10, 100)
    populations = list(map(lambda x: x.census(), populations))
    # print(np.asarray(populations[0]))
    # print(np.asarray(populations))
    make_film(lambda x: x**2, populations, step=20)
