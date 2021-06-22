import numpy as np
from random import random, randint, uniform, choice, choices, sample
from tqdm import tqdm
from time import time
import sys
from datetime import datetime
from os import makedirs
import traceback
import shutil
from getpass import getpass
import smtplib
import logging
from keras.utils import to_categorical
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from math import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf

class Classifier:
    def __init__(self, x_train = None, y_train = None, x_val = None, y_val = None,
                 x_test = None, y_test = None, verbose = 1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.verbose = verbose

        self.num_classes = 10
        self.height = self.x_train.shape[1]
        self.width = self.x_train.shape[0]
        self.channels = 1

        self.model = None


    def fit(self, is_test = False, batch_size = 64, epochs = 10):

        if not is_test:
            self.model.fit(self.x_train[:,14-int(self.w_input/2):14+int(self.w_input/2),14-int(self.h_input/2):14+int(self.h_input/2),:], self.y_train, batch_size=batch_size,
                           epochs=epochs, verbose=self.verbose)
        else:
            self.model.fit(self.x_train[:,14-int(self.w_input/2):14+int(self.w_input/2),14-int(self.h_input/2):14+int(self.h_input/2),:], self.y_train, batch_size=batch_size,
                           epochs=epochs, verbose = self.verbose,
                           validation_data = (self.x_val[:,14-int(self.w_input/2):14+int(self.w_input/2),14-int(self.h_input/2):14+int(self.h_input/2),:], self.y_val))


    def clear(self):
        K.clear_session()
        tf.compat.v1.reset_default_graph()





    def evaluate(self, is_test, batch_size = 64):
        if not is_test:
            scores = self.model.evaluate(self.x_val[:,14-int(self.w_input/2):14+int(self.w_input/2),14-int(self.h_input/2):14+int(self.h_input/2),:],
                                         self.y_val,batch_size = batch_size,verbose = self.verbose)
        else:
            scores = self.model.evaluate(self.x_test[:,14-int(self.w_input/2):14+int(self.w_input/2),14-int(self.h_input/2):14+int(self.h_input/2),:],
                                         self.y_test, batch_size=batch_size, verbose=self.verbose)
        return dict(zip(self.model.metrics_names, scores))


    def model_architecture_config(self, parameters):
        self.model = Sequential()
        print(parameters)
        self.w_input = self.x_train.shape[1] - 2 * parameters['active_inputs_w']
        self.h_input = self.x_train.shape[2] - 2 * parameters['active_inputs_h']
        #print(self.x_val.shape)
        self.model.add(Flatten(input_shape=(self.w_input, self.h_input)))
        # FULLY CONNECTED LAYERS
        self.model.add(Dense(397, activation='relu', ))
        self.model.add(Dense(397, activation='relu'))
        # OUTPUT LAYER
        self.model.add(Dense(self.num_classes, activation='softmax'))

        opt = Adam(learning_rate=0.001)



        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


class Genetic_Algorithm:
    def __init__(self, parameters, fitness_function, population_size, generations,
                 elitism = 0.1, crossover_rate = 0.8, crossover_point = None,
                 mutation_rate = 0.25, random_selection_rate = 0.01):
        self.elitism = elitism
        self.generations = generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_selection_rate = random_selection_rate

        self.parameters_range = list(parameters.values())
        self.fitness_function = fitness_function
        if crossover_point == None:
            self.crossover_point = int(len(parameters)/2)
        else:
            self.crossover_point = crossover_point
        self.precision = 7


    def create_individual(self):
        '''
        Individual is represented as a possible solution
        to the problem.

        In this case a solution is an array with values of
        the selected hyperparameters.

        A probability distribution (random, gaussian, uniform) is the best way to generate
        values inside a range of possible values
        '''
        return [round(uniform(*parameter), 3) if type(parameter) == tuple else choice(parameter)
                for parameter in self.parameters_range]

    def individual_format(self, individual):
        print(individual)
        return tuple(individual)

    def create_population(self):
        '''
        Create an initial random population according with the
        parameters of the problem and its valid values
        '''
        print("Creating initial random population...")
        population = []
        while len(population) < self.population_size:
            ind = self.create_individual()
            print(ind)
            population.append(ind)
        return population

    #   FITNESS SECTION
    def fitness(self, individual):
        '''
        function fitness = evaluate_individual
        '''
        ind = self.individual_format(individual)
        return self.fitness_function(ind)

    def sort_by_fitness(self, population):
        scores = [self.fitness(individual) for individual in
                  tqdm(population, desc="Measuring Population Fitness", file=sys.stdout)]
        return [x for _, x in sorted(zip(scores, population), key=lambda p: p[0], reverse=True)]

    def population_fitness(self, population):
        return [self.fitness(individual) for individual in tqdm(population, desc="Measuring Population Fitness", file=sys.stdout)]

    def order_population(self, scores, population):
        self.scores, self.population = [list(t) for t in zip(*sorted(zip(scores, population)))]

    def grade(self, list_fit=None):
        '''
        Find minimum fitness for a population.
        '''
        if not list_fit:
            list_fit = self.scores
        try:
            return np.nanmin([fit for fit in self.scores])
        except:
            return np.nan

    # REPRODUCTION SECTION
    def crossover(self, individual1, individual2):

        child1 = individual1.copy()
        child2 = individual2.copy()

        if np.random.uniform(0, 1) < self.crossover_rate:
            child1 = individual1[:self.crossover_point] + individual2[self.crossover_point:]
            child2 = individual2[:self.crossover_point] + individual1[self.crossover_point:]

        return child1, child2

    # MUTATION SECTION
    def mutation(self, individual):
        if np.random.uniform(0, 1) < self.mutation_rate:
            locus = randint(0, len(individual) - 1)
            parameter = self.parameters_range[locus]
            individual[locus] = uniform(*parameter) if type(parameter) == tuple else choice(parameter)

    # GENERATIONAL SECTION
    def evolve(self):

        # ELITISM
        elitism_size = int(self.population_size * self.elitism)
        # orderedPop = self.sortByFitness(population)
        new_generation = [ind for ind in
                         tqdm(self.population[:elitism_size], desc="Applying Elitism", file=sys.stdout)]

        while len(new_generation) < self.population_size:

            # RANDOM SELECTION (DIVERSITY)
            for individual in tqdm(self.population[elitism_size:], desc="Random Selection", file=sys.stdout):
                if np.random.uniform(0, 1) < self.random_selection_rate:
                    new_generation.append(individual)

            # RANDOM MUTATION (DIVERSITY)
            for individual in tqdm(self.population[elitism_size:], desc="Random Mutation", file=sys.stdout):
                self.mutation(individual)
                new_generation.append(individual)

            # CROSSOVER
            ind1, ind2 = sample(self.population, 2)

            child1, child2 = self.crossover(ind1, ind2)

            if np.random.uniform(0, 1) < self.mutation_rate:
                random_selection = choice([child1, child2])
                self.mutation(random_selection)
                new_generation.append(random_selection)
            new_generation.append(child1)
            new_generation.append(child2)

        # EVALUATE POPULATION
        generation_scores = self.population_fitness(new_generation)
        generation_best_fitness = self.grade(generation_scores)

        print("Best fitness of this generation:", generation_best_fitness)

        self.order_population(generation_scores, new_generation)
        self.best_fitness = generation_best_fitness


    def population_info(self, population):
        pass


    def run(self):

        counter = 0
        # CREATE INITIAL RANDOM POPULATION
        self.population = self.create_population()

        # EVALUATE INITIAL POPULATION
        self.scores = self.population_fitness(self.population)
        self.best_fitness = self.grade()
        print("Initial best fitness:", self.best_fitness)

        # ORGANIZING POPULATION BY FITNESS
        self.order_population(self.scores, self.population)

        while counter < self.generations:
            print(f"\n  Running iteration {(counter + 1)}/{self.generations}")

            self.evolve()

            counter += 1

        return self.best_fitness, self.population


# AUXILIARY FUNCTIONS
def convert_pow2(num):
    '''
      Convert num to the closest power of 2
    '''
    return int(pow(2, ceil(log2(abs(num)))))


def convert_range(num, bounds):
    '''
      Clip number to the bounds
    '''
    num_digits = 2
    num = round(abs(num), num_digits)
    return np.clip(num, *bounds)


def load_dataset():
    (train_digits, train_labels), (test_digits, test_labels) = load_data()
    return (train_digits, train_labels), (test_digits, test_labels)


def load_dataset_with_validation(rate=0.10):
    """
    Load dataset setting apart some validation data
    @args:
        - rate: Percentage of training data to validation
    """

    (train_digits, train_labels), (test_digits, test_labels) = load_dataset()

    # RESHAPE DATA
    train_data = reshape_dataset(train_digits)
    test_data = reshape_dataset(test_digits)

    # RESCALE DATA
    train_data = rescale_dataset(train_data)
    test_data = rescale_dataset(test_data)

    # ONE-HOT ENCODING
    train_labels_cat = encoding_dataset(train_labels)
    test_labels_cat = encoding_dataset(test_labels)

    # SHUFFLE THE TRAINING DATASET
    for _ in range(5):
        indexes = np.random.permutation(len(train_data))

    train_data = train_data[indexes]
    train_labels_cat = train_labels_cat[indexes]

    split_point = int(rate * len(train_data))

    validation_data = train_data[:split_point, :]
    validation_labels_cat = train_labels_cat[:split_point, :]

    train_data2 = train_data[split_point:, :]
    train_labels_cat2 = train_labels_cat[split_point:, :]

    return train_data2, train_labels_cat2, test_data, test_labels_cat, validation_data, validation_labels_cat


def reshape_dataset(data):
    """
    Reshaping data to CNN standard
    """
    height = data.shape[1]
    width = data.shape[2]
    channels = 1

    return np.reshape(data, (data.shape[0], height, width, channels))


def rescale_dataset(data):
    """
    Rescaling data
    """
    return data.astype('float32') / 255


def encoding_dataset(data_labels, num_classes=10):
    """
    ONE-HOT ENCODING
    @args:
        - dataLabels
        - numClasses

    @output:
        - List of classes
    """
    return to_categorical(data_labels, num_classes)


def show_random_images(data, labels):
    """
    Exhibit 14 random samples from dataset
    """

    np.random.seed(123)

    rand_14 = np.random.randint(0, data.shape[0], 14)
    sample_digits = data[rand_14]
    sample_labels = labels[rand_14]

    num_rows, num_cols = 2, 7

    f, ax = plt.subplots(num_rows, num_cols, figsize=(12, 5),
                         gridspec_kw={'wspace': 0.03, 'hspace': 0.01},
                         squeeze=True)

    for r in range(num_rows):
        for c in range(num_cols):
            image_index = r * 7 + c
            ax[r, c].axis("off")
            ax[r, c].imshow(sample_digits[image_index], cmap='gray')
            ax[r, c].set_title('No. %d' % sample_labels[image_index])
    plt.show()


def run(**kwargs):
    algorithm = kwargs.get('algorithm')
    dataset = kwargs.get('dataset')

    if kwargs.get('algorithm') == 'GA':
        parameters = kwargs.get('parameters')
        pop_size = kwargs.get('population_size')
        generations = kwargs.get('generations')
        fitness = kwargs.get('fitness')
        # evolver = GA(fitness, parameters, popSize, generations, history)
        evolver = Genetic_Algorithm(parameters=parameters, fitness_function=fitness, population_size=pop_size,
                                   generations=generations)
    else:
        pass

    best, population = evolver.run()
    print("Best Solution after " + str(generations) + " generations...")
    print(
        "\n active_inputs_w: " + str(population[0][0]),
        "\n active_inputs_h: " + str(population[0][1])

    )
    print("Fitness (loss)" + str(best))


def main(parameters, BATCH_SIZE, EPOCHS, GENERATIONS, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE):
    train_data, train_labels_cat, \
    test_data, test_labels_cat, \
    validation_data, validation_labels_cat = load_dataset_with_validation()
    print(train_data.shape)
    # Instantiate CNN Clssifier with the MNIST dataset
    cnn = Classifier(x_train=train_data, y_train=train_labels_cat,
                     x_val=validation_data, y_val=validation_labels_cat,
                     x_test=test_data, y_test=test_labels_cat,
                     verbose=1)

    def fitness(individual, test=False):
        cnn.clear()

        cnn.model_architecture_config(dict(zip(parameters.keys(), individual)) if not isinstance(individual, dict) else individual)

        cnn.fit(batch_size=BATCH_SIZE, epochs=EPOCHS)

        results = cnn.evaluate(test)
        results['loss'] = results['loss']*np.sqrt((28-2*individual[0])*((28-2*individual[1])))
        print(results)
        return results['loss']

    run(algorithm='GA', dataset='NMIST', fitness=fitness, parameters=parameters,
        population_size=POPULATION_SIZE, generations=GENERATIONS)


if __name__ == '__main__':
    # GLOBAL GA PARAMETERS
    GENERATIONS = 10
    POPULATION_SIZE = 20
    MUTATION_RATE = 0.01
    CROSSOVER_RATE = 0.1

    # GLOBAL CNN PARAMETERS
    EPOCHS = 10
    BATCH_SIZE = 256

    # Hyperparameters:
    # - Learning rate

    # Fitness:
    # - loss
    # - accuracy
    active_inputs = []
    for i in range(0,784):
        active_inputs.append(i)
    parameters = {
        'active_inputs_w': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'active_inputs_h': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    }
    main(parameters,BATCH_SIZE,EPOCHS,GENERATIONS,POPULATION_SIZE,MUTATION_RATE,CROSSOVER_RATE)