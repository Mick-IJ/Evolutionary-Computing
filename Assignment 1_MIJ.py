import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller
from random import sample

ENV = Environment(experiment_name="test",
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemy_controller=enemy_controller(),
                  level=2,
                  speed="fastest",
                  contacthurt='player',
                  logs='off')


class Individual:
    dom_u = 1
    dom_l = -1
    mutation_rate = .3
    n_hidden = 10
    n_vars = (ENV.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5  # multilayer with 50 neurons

    def __init__(self):
        self.age = 0
        self.weights = list()
        self.fitness = None
        self.enemy_life = None
        self.player_life = None
        self.time = None
        self.children = 0

    def set_weights(self, weights=None, dom_l=dom_l, dom_u=dom_u, n_vars=n_vars):
        if weights is None:
            self.weights = np.random.uniform(dom_l, dom_u, n_vars)
        else:
            self.weights = weights

    def evaluate(self):
        f, p, e, t = ENV.play(pcont=self.weights)
        self.fitness = f
        self.player_life = p
        self.enemy_life = e
        self.time = t

    def check_limits(self, dom_u=dom_u, dom_l=dom_l):
        new_weights = list()
        for weight in self.weights:
            if weight > dom_u:
                new_weights.append(dom_u)
            elif weight < dom_l:
                new_weights.append(dom_l)
            else:
                new_weights.append(weight)

        self.weights = np.asarray(new_weights)

    def birthday(self):
        self.age += 1

    def mutate(self, mutation_rate=mutation_rate):
        for i in range(0, len(self.weights)):
            if np.random.random() <= mutation_rate ** 2:
                self.weights[i] = np.random.normal(0, 1)
            if np.random.random() <= mutation_rate:
                self.weights[i] = self.weights[i] * np.random.normal(0, 1.27)
            if np.random.random() <= mutation_rate:
                self.weights[i] = self.weights[i] + np.random.normal(0, .1)

        self.check_limits()


class Population:
    max_children = 1

    def __init__(self, size=5):
        self.individuals = list()
        self.size = size
        self.mean_fit = None
        self.max_fit = None
        self.mean_age = None
        self.mean_children = None
        self.generation = 1
        self.mutation_rate = .3
        self.mean_fit_history = list()
        self.max_fit_history = list()

    def append(self, individual):
        self.individuals.append(individual)
        self.update_stats()

    def extend(self, population):
        self.individuals.extend(population)
        self.update_stats()

    def kill(self, individual):
        self.individuals.remove(individual)
        self.update_stats()

    def update_stats(self):
        population_fit = [i.fitness for i in self.individuals]
        self.mean_fit = np.mean(population_fit)
        self.max_fit = np.max(population_fit)
        self.mean_age = np.mean([i.age for i in self.individuals])
        self.mean_children = np.mean([i.children for i in self.individuals])

    def display_population(self):
        i = 1
        for individual in self.individuals:
            print(f'{i}: fitness = {round(individual.fitness, 4)}, age = {individual.age}',
                  f'children = {individual.children}')
            i += 1

        print(f'Mean fitness: {round(self.mean_fit, 4)}, Mean age: {self.mean_age}',
              f'Mean children = {self.mean_children}\n')

    def initialize(self):
        for i in range(self.size):
            individual = Individual()
            individual.set_weights()
            individual.evaluate()
            individual.birthday()
            self.individuals.append(individual)

        self.update_stats()
        self.display_population()

    def select_parents(self, n, type_='random'):
        if type_ == 'random':
            return sample(self.individuals, n)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            probabilities = [(fit/sum(pop_fitness)) for fit in pop_fitness]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)

    def trim(self):
        pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
        probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]

        self.individuals = list(np.random.choice(self.individuals, size=self.size, replace=False, p=probabilities))
        self.update_stats()

    def sex(self, max_children=max_children, type_='mean'):

        parent1, parent2 = self.select_parents(2, type_='fit')

        for i in range(max_children):
            cross_prop = np.random.random()

            if type_ == 'mean':
                child_weights = np.array(parent1.weights) * cross_prop + np.array(parent2.weights) * (1 - cross_prop)

            else:  # type_ == 'recombine':
                split_loc = int(len(parent1.weights) * cross_prop)
                child_weights = np.append(parent1.weights[:split_loc], parent2.weights[split_loc:])

            child = Individual()
            child.set_weights(child_weights)
            child.mutate(mutation_rate=self.mutation_rate)
            child.evaluate()
            self.individuals.append(child)

            parent1.children += 1
            parent2.children += 1

    def mutation_rate_change(self, type_='exp'):
        if type_ == 'linear':
            self.mutation_rate -= 0.01
        elif type_ == 'exp':
            self.mutation_rate *= 0.99
        elif type_ == 'log':
            self.mutation_rate = np.log(self.mutation_rate)

    def next_generation(self):
        self.generation += 1
        for individual in self.individuals:
            individual.birthday()
        self.mean_fit_history.append(self.mean_fit)
        self.max_fit_history.append(self.max_fit)

    def plot_generations(self):
        plt.figure(figsize=(12, 12))
        plt.plot(self.max_fit_history, label="best")
        plt.plot(self.mean_fit_history, label="avg")
        plt.ylim((0, 100))
        plt.legend()
        plt.title("First run score")
        plt.show()


def main(size=5, generations=5, children_per_gen=5):
    population = Population(size)
    population.initialize()

    for i in range(generations):
        print('Generation:', population.generation)

        for j in range(children_per_gen):
            population.sex(type_='mean')

        population.trim()
        population.display_population()

        population.mutation_rate_change(type_='exp')
        population.next_generation()

    population.plot_generations()


main(size=10, generations=7, children_per_gen=8)
