import copy

import numpy as np
import math
import os

from animator import basic_func, objects


def make_polynomial(coef):
    """
    Return Python function from polynomial coefficients.
    Args:
        coef (np.array): List of coefficients.

    Returns:
        func: Function representing polynomial.
    """
    return lambda x: sum(a*x**i for i, a in enumerate(coef))


class Approximant:
    """
    Class representing single polynomial approximant.

    Attributes:
        length (int): Polynomial's degree.
        values (np.array): Polynomial's coefficients.
        std (np.array): Vector of standard deviations for each coefficient used in mutation process.
        temp_fitness_rank (int): Saving temporary rank of fitness in Population.
        temp_diversity_rank (int): Saving diversity rank of fitness in Population.
    """
    def __init__(self, length, values=None, std=None, default_std=1):

        self.length = length
        self.values = np.random.normal(0, default_std, (length,))
        self.std = np.abs(np.random.normal(0, default_std, (length,)))
        if values is not None:
            self.values = values
            self.std = std
        self.temp_fitness_rank = None
        self.temp_diversity_rank = None

    def make_polynomial(self):
        """
        Making polynomial function out of coefficients.

        Returns:
            func: Polynomial function.
        """
        return make_polynomial(self.values)

    def sex(self, other):
        """
        Averages two polynomials producing a successor.
        Args:
            other (Approximant): Another parent polynomial

        Returns:
            Approximant: Baby polynomial.
        """
        t = np.random.random()
        return Approximant(self.length, values=t*self.values + (1-t)*other.values,
                           std=t*self.std + (1-t)*other.std, )

    def mutate(self):
        """
        Mutates itself with respect to own standard deviation vector.

        Returns:
            self
        """
        self.values += np.random.normal(0, self.std, (self.length,))
        self.std += np.random.normal(np.zeros((self.length,)), self.std, (self.length,))
        self.std = np.abs(self.std)
        return self

    def fitness(self, target_foo, norm='sup', sampling_rate=100):
        """
        Computes fitness with respect to given norm and target function.

        Args:
            target_foo (func): Function to be compared with.
            norm: Type of norm. Two norms are possible:
                'int': Integral from 0 to 1 from absolute value of difference between functions.
                'sup': Supremum of absolute value of difference between functions on [0, 1]
            sampling_rate (int): Precision of norm calculation.

        Returns:
            float: Fitness value.
        """
        if norm == 'sup':
            return 1/(max([abs(target_foo(x) - self.make_polynomial()(x)) for x in np.linspace(0, 1, sampling_rate)])+1)
        if norm == 'int':
            return 1/(sum([abs(target_foo(x) - self.make_polynomial()(x)) for x in np.linspace(0, 1, sampling_rate)])
                      / sampling_rate + 1)

    def diversity(self, new_population):
        """
        Calculates the distance between this polynomial and family of polynomials stored in Population.

        Args:
            new_population: Calculating the distance to this family of polynomials.

        Returns:
            float: Diversity value.
        """
        return np.mean(np.array([np.linalg.norm(self.values - x.values) for x in new_population]))

    def __str__(self):
        """
        Converting polynomial coefficients to str.

        Returns:
            str: Polynomial coefficients.
        """
        return ' '.join(map(str, self.values))


class Population:
    """
    Represents family of polynomial functions.

    Attributes:
        size (int): Family's cardinality.
        p_c (float): Probability of choosing the best fitting polynomial during ranking selection.
        unit_length (int): Degree of single polynomial.
        society (list): List of polynomials.
        target_function (func): Function to be approximated.
        metric (str): Type of metric used in calculating fitness. Described in Approximant.fitness docstring.
        default_std (float): Standard deviation of first random choices.
    """
    def __init__(self, size, unit_length, target_function, society=None, default_std=1, p_c=.1, metric='int'):
        self.size = size
        self.p_c = p_c
        self.unit_length = unit_length
        self.society = society if society is not None else [Approximant(unit_length, default_std=default_std) for _ in range(size)]
        self.target_function = target_function
        self.metric = metric
        self.default_std = default_std

    def group_sex(self, selection_type='proportional', save_king=True):
        """
        Creating new generation out of this one by averaging, mutating and selecting new polynomials.

        Args:
            selection_type (str): Selection algorithm. There are 3 implemented algorithms for selection:
                'proportional': Probability of selection is proportional to it's fitness.
                'rank': Probability of selection is based on rank in fitness ranking and parameter p_c.
                'diversity rank': Probability of selection every new polynomial is based on it's
                    rank in fitness ranking and distance between already selected family of polynomials.
            save_king (bool): If True, the best polynomial will be automatically placed in new generation.

        Returns:
            Population: New generation of polynomials.
        """
        self.society.sort(key=lambda x: x.fitness(self.target_function, norm=self.metric), reverse=True)

        for i, x in enumerate(self.society):
            x.temp_fitness_rank = i

        if selection_type == 'diversity rank':
            distribution = [(1 - self.p_c) ** k * self.p_c for k in range(self.size - 1)] + \
                           [(1 - self.p_c) ** (self.size - 1)]
            parents = np.random.choice(self.society, 2, replace=True, p=distribution)
            new_population = [parents[0].sex(parents[1]).mutate()]

            while len(new_population) != self.size:
                self.society.sort(key=lambda guy: guy.diversity(new_population), reverse=True)
                for i, x in enumerate(self.society):
                    x.temp_diversity_rank = i
                self.society.sort(key=lambda guy: guy.temp_fitness_rank**2 + guy.temp_diversity_rank**2)
                parents = np.random.choice(self.society, 2, replace=True, p=distribution)
                new_population += [parents[0].sex(parents[1]).mutate()]
                if save_king:
                    new_population[-1] = max(self.society, key=lambda x: x.fitness(self.target_function,
                                                                                   norm=self.metric))
            return Population(self.size, self.unit_length, self.target_function, new_population, metric=self.metric)

        if selection_type == 'proportional':
            sum_of_fitness = sum([x.fitness(self.target_function, norm=self.metric) for x in self.society])
            distribution = np.array([x.fitness(self.target_function, norm=self.metric)/sum_of_fitness for x in self.society])

        else:
            # 'rank' selection algorithm.
            distribution = np.array([(1-self.p_c)**k*self.p_c for k in range(self.size-1)]
                                    + [(1-self.p_c)**(self.size-1)])

        new_mothers = np.random.choice(np.array(self.society), self.size, replace=True, p=distribution)
        new_fathers = np.random.choice(self.society, self.size, replace=True, p=distribution)
        new_population = [x.sex(y).mutate() for x, y in zip(new_fathers, new_mothers)]
        if save_king:
            new_population[0] = max(self.society, key=lambda x: x.fitness(self.target_function, norm=self.metric))
        return Population(self.size, self.unit_length, self.target_function, new_population, metric=self.metric)

    def census(self):
        """
        Returns list of sorted polynomials as python functions.

        Returns:
            list: Sorted list of polynomials
        """
        self.society.sort(key=lambda x: x.fitness(self.target_function, norm=self.metric), reverse=True)
        return [make_polynomial(x.values) for x in self.society]

    def get_best_fitness(self):
        """
        Calculating the best fitness of polynomials from this family.

        Returns:
            float: Best fitness.
       """
        self.society.sort(key=lambda guy: guy.fitness(self.target_function, norm=self.metric), reverse=True)
        return self.society[0].fitness(self.target_function, norm=self.metric)

    def __str__(self):
        """
        Returning coefficients of polynomials from this family in string.

        Returns:
            str: Coefficients listed in string.
        """
        return '\n'.join(map(str, self.society))

    def change_unit_length(self, n):
        """
        Changing the polynomials degree to higher one by drawing randomly missing coefficients.

        Args:
            n (int): Target degree
        """
        self.unit_length = n
        for guy in self.society:
            values = np.random.normal(0, self.default_std, (n,))
            std = np.abs(np.random.normal(0, self.default_std, (n,)))
            values[:guy.values.shape[0]] = guy.values
            std[:guy.std.shape[0]] = guy.std
            guy.values = values
            guy.std = std
            guy.length = n

    def add_new(self, n):
        """
        Drawing n new random polynomials at the end of society.

        Args:
            n (int): Number of new polynomials.
        """
        self.society[-n:] = [Approximant(self.unit_length, default_std=self.default_std) for _ in range(n)]


def genetic_algorithm(target_function, population_size, unit_length, epochs, selection_type='rank', default_std=1,
                      save_king=True, p_c=.1, metric='int', starting_population=None):
    """
    Final genetic algorithm run.

    Args:
        target_function (func): Function to be approximated.
        population_size (int): Size of single polynomial family.
        unit_length (int): Degree of single polynomial.
        epochs (int): Number of generations.
        selection_type (str): Selection algorithm. Described in Population.group_sex docstring.
        default_std (float): Initial standard deviation.
        save_king (bool): If True best polynomial from each generation will be saved to the next one.
        p_c (float): Probability of choosing best polynomial in rank selection.
        metric (str): Type of metric.
        starting_population (Population or None): Initial Population. If None, population will be generated randomly.

    Returns:
        list: List of populations.
    """
    if starting_population is None:
        populations = [Population(population_size, unit_length, target_function, default_std=default_std, p_c=p_c,
                                  metric=metric)]
    else:
        populations = [starting_population]

    for i in range(epochs):
        populations.append(populations[-1].group_sex(selection_type=selection_type, save_king=save_king))
        print(f'{i}: {populations[-1].get_best_fitness()}')
    return populations


def learning_curve(populations, filename='learning_curve.png', inverse=False, metric='sup'):
    """
        Only for learning curve generation. Don't worry about it.
    """
    if inverse:
        points = list(map(lambda n: (n[0], 1/n[1].get_best_fitness() - 1), tuple(enumerate(populations))))
    else:
        points = list(map(lambda n: (n[0], n[1].get_best_fitness()), tuple(enumerate(populations))))

    frame = basic_func.OneAxisFrame((1920, 1080), 'black', 100, 100)
    func = objects.PolygonalChain(points)

    settings_function = {
        'sampling rate': 30,
        'thickness': 5,
        'blur': 2,
        'color': 'gray'
    }
    settings_axes = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }
    settings_grid = {
        'sampling rate': 3,
        'thickness': 5,
        'blur': 2,
        'color': 'white'
    }

    if inverse and metric == 'sup':
        frame.add_axis_surface(x_bounds=(-5, len(populations)), y_bounds=(-.55, 2.05))
        frame.blit_axes(settings_axes, x_only=False)
        frame.blit_x_grid(settings_grid, interval=len(populations) / 20, length=.014)
    else:
        frame.add_axis_surface(x_bounds=(-5, len(populations)), y_bounds=(-.55, 1.05))
        frame.blit_axes(settings_axes, x_only=False)
        frame.blit_x_grid(settings_grid, interval=len(populations) / 20, length=.007)

    frame.blit_y_grid(settings_grid, interval=.25, length=len(populations)/500)
    frame.axis_surface.blit_parametric_object(func, settings_function, interval_of_param=(0, len(populations)-1.01))
    frame.blit_axis_surface()
    frame.generate_png(filename)


def epoch_frame(target_func, approximators_list, generate_png=False, top_n=5):
    """
    Plotting family of functions belonging to single epoch and the target function.

    Args:
        target_func (function): Function to be approximated.
        approximators_list (list of functions): Approximators from one epoch.
        generate_png (bool, optional): Saves the image to 'frame.png' if True.
        top_n (int): How many best functions should be printed.

    Returns:
        basic_func.OneAxisFrame: Generated frame.
    """
    frame = basic_func.OneAxisFrame((1280, 720), 'black', 50, 50)
    target_function = objects.Function(target_func)
    approximators = [objects.Function(foo) for foo in approximators_list[:top_n]]

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
    frame.blit_axes(settings_axes, x_only=True)
    for foo in approximators:
        frame.axis_surface.blit_parametric_object(foo, settings_approximators, queue=True)
    frame.axis_surface.blit_parametric_queue()
    frame.blit_parametric_object(target_function, settings_target_function)
    frame.blit_axis_surface()
    if generate_png:
        frame.generate_png(f'frame.png')
    return frame


def make_film(target_func, epochs, filename='genetic.mp4', fps=1, resolution=(1280, 720), step=1, top_n=5,
              number_of_frames=5, save_ram=True, id='', read_only=False):
    """
    Generates the video illustrating function approximation by genetic algorithm.

    Args:
        target_func (function): Function to be approximated.
        epochs (list): List of epochs, where epoch is a list of approximators belonging to that epoch.
        filename (str): Target filename.
        fps (int): Frames per second.
        resolution: Resolution of the video. Should be the same as frame resolution.
        step (int): Skip every step epochs in the video.
        top_n (int): How many best function should be plotted.
        number_of_frames (int): IF not None, step is to generate specific number of frames
        save_ram (bool): Render in save_ram mode.
        id (str): Film id.
        read_only (bool): Render already saved frames.
    """

    video = basic_func.Film(fps, resolution, id=id)

    if number_of_frames is not None:
        number_of_frames -= 1

    if read_only:
        video.frame_counter = number_of_frames
        video.render(filename, save_ram=True)
        return

    if number_of_frames is not None:
        step = len(epochs)//number_of_frames

    print(f'step: {step}, frames: {len(epochs[::step])}')
    for i, epoch in enumerate(epochs[::step]):
        video.add_frame(epoch_frame(target_func, epoch, top_n=top_n), save_ram=save_ram)
        print(i)
    video.render(filename, save_ram=save_ram)


def generate_curve(pop_size, unit_len, epochs, selection, std, pc, metric, inverse=False, filename=None):
    """
        Only for learning curve generation. Don't worry about it.
    """
    basic_func.DEBUG = True
    target = lambda x: math.sin(10 * x)
    populations_ = genetic_algorithm(target, population_size=pop_size, unit_length=unit_len, epochs=epochs,
                                     selection_type=selection, default_std=std, save_king=True, p_c=pc, metric=metric)
    if filename is None:
        learning_curve(populations_, filename=f'lc_size{pop_size}_len{unit_len}_ep{epochs}_std{std}_pc{pc}_{metric}.png'
                       , inverse=inverse, metric=metric)
    else:
        learning_curve(populations_, filename=filename, inverse=inverse, metric=metric)


def sequential(target_function, deg, population_size=100, epochs=200,
               selection_type='rank', default_std=4, save_king=True, p_c=.1, metric='sup', start=5, add_new=50):
    """
    Algorithm firstly approximating lower degree polynomials. To be tested.
    """
    print(f'deg: {start}')
    pop = genetic_algorithm(target_function, population_size, start, epochs, selection_type, default_std, save_king,
                                  p_c, metric)
    for n in range(start+1, deg+1):
        print(f'deg: {n}')
        new_start = copy.copy(pop[-1])
        new_start.change_unit_length(n)
        new_start.add_new(add_new)
        pop += genetic_algorithm(target_function, population_size, n, epochs, selection_type, default_std, save_king,
                                 p_c, metric, starting_population=new_start)
    return pop


if __name__ == '__main__':
    # basic_func.DEBUG = True
    # init()

    # generate_curve(pop_size=50, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    # generate_curve(pop_size=100, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    # generate_curve(pop_size=200, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    # generate_curve(pop_size=350, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)
    # generate_curve(pop_size=500, unit_len=15, epochs=1000, selection='rank', std=5, pc=.5, metric='sup', inverse=True)

    target = lambda x: math.sin(10*x)
    populations_ = sequential(target, population_size=100, deg=20, epochs=100,
                                     selection_type='rank', default_std=4, save_king=True, p_c=.35, metric='int', start=1, add_new=50)
    learning_curve(populations_, filename='lc_sup_l10.png', inverse=True)
    populations_ = list(map(lambda x: x.census(), populations_))
    make_film(target, populations_, filename='sup_genetic_l15.mp4', fps=15, resolution=(1280, 720), step=1, top_n=5,
              number_of_frames=150, save_ram=True, id='_gn3_', read_only=False)


