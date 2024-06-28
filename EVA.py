from settings import Settings
import math
import numpy as np

set = Settings()


class Eva:
    """class for Evolutionary Algorithm (EVA)"""

    def __init__(self):
        self.max_fitness = 0
        self.best_ind = []


node_num = set.length
POP_SIZE = set.POP_SIZE
DNA_SIZE = set.DNA_SIZE
MUTATION_RATE = set.MUTATION_RATE


def fillBits(size):
    return 1 << size - 1


def target_function():
    """set sine function as target function"""
    """
    # define the constants
    blen = set.beam_length
    sep_x = (set.screen_width - (set.row_lenh - 1) * blen * math.sqrt(3)) / 2
    sep_y = (set.screen_height - ((set.row_num - 1) / 2) * blen) / 2

    # define the sine function related parameters
    T = set.beam_length * 2 * math.sqrt(3) * 1.02
    omiga = 2 * math.pi / T
    Amp = set.beam_length / 25
    bias = set.screen_height - sep_y + 12
    return lambda x: Amp * math.sin(omiga * (x - sep_x) + math.pi / 2) + bias
    """

    # define the constants
    blen = set.beam_length
    sep_x = (set.screen_width - (set.row_lenh - 1) * blen * math.sqrt(3)) / 2
    sep_y = (set.screen_height - ((set.row_num - 1) / 2) * blen) / 2

    # define the sine function related parameters
    T = set.row_lenh * blen * math.sqrt(3)
    omiga = 2 * math.pi / T
    Amp = set.beam_length / 15
    # bias = set.screen_height - sep_y - blen / 2 + blen / 3
    bias = set.screen_height - sep_y + 5
    return (
        lambda x: Amp * math.sin(omiga * (x - sep_x + blen * math.sqrt(3) / 2)) + bias
    )


def avoid_function_lin():
    """set the function that the input nodes should avoid"""
    # type1 linear function
    sep_y = (set.screen_height - ((set.row_num - 1) / 2) * set.beam_length) / 2
    Amp = set.beam_length / 2
    bias = Amp / 3
    return lambda x: sep_y + bias


def avoid_function_sin():
    """set the function that the input nodes should avoid"""
    # type2 sine function
    T = set.screen_width
    omiga = 2 * math.pi / T
    Amp = set.beam_length / 2
    bias = Amp / 4
    return lambda x: Amp * math.sin(omiga * x + math.pi) + bias


def rms_func(rms):
    """process the root mean square error"""
    # approximate the Dirac delta function
    a = 4
    gauss = math.exp(-(rms**2) / (a**2)) / (a * math.sqrt(math.pi))
    return gauss


def get_fitness(indPos):
    """calculate the fitness of a certain individual"""
    target = target_function()
    length = set.row_lenh
    num = node_num - 1
    sum_output = 0

    for i in range(length):
        bias_out = (target(indPos[num - i][0]) - indPos[num - i][1]) ** 2
        sum_output += bias_out

    # rms_out = math.sqrt(sum_output / length)
    # fitness = rms_func(rms_out)
    # fitness = 1 / (math.exp(rms_out) + 1)

    fitness = max(1e-5, 100 - sum_output)
    # revised_fitness = math.exp(fitness / 100) * 100 / math.exp(1)

    return fitness


def select_parent(pop, fitness):
    """choose the parent based on fitness"""
    index = np.random.choice(
        POP_SIZE, size=POP_SIZE, replace=True, p=fitness / sum(fitness)
    )

    return [pop[index[i]] for i in range(POP_SIZE)]


def crossover(pop, parent, crossover_rate, fitness, type=0):
    """crossover the parents to generate offspring"""
    if np.random.rand() < crossover_rate:

        # index = np.random.choice(POP_SIZE, p=fitness / sum(fitness))
        # point = np.random.randint(1, DNA_SIZE - 1)

        # if type == 0:
        #     crossover_result = [
        #         pop[index][i] if abs(i - point) < 2 else parent[i]
        #         for i in range(node_num)
        #     ]

        # if type == 1:
        #     crossover_result = [
        #         np.concatenate((parent[i][:point], pop[index][i][point:]))
        #         for i in range(node_num)
        #     ]

        # return crossover_result

        # temp = pop.copy()
        point_num = int(DNA_SIZE / 10) + 1
        point = np.random.choice(
            np.arange(1, DNA_SIZE - 1), replace=False, size=point_num
        )
        index = np.random.choice(POP_SIZE, p=fitness / sum(fitness))

        for k in range(point_num):

            if type == 0:
                pop = [
                    pop[index][i] if abs(i - point[k]) < 2 else parent[i]
                    for i in range(node_num)
                ]

            if type == 1:
                pop = [
                    np.concatenate((parent[i][: point[k]], pop[index][i][point[k] :]))
                    for i in range(node_num)
                ]

            return pop

    else:
        return parent


def mutate(child, mutation_rate):
    """mutation operator"""
    if np.random.rand() < mutation_rate:
        child = np.array(child)
        interval = np.max(child) - np.min(child)
        point = np.random.randint(0, DNA_SIZE)

        # mutation process
        child = interval / (1 + child) + (point + 1) * np.random.rand(
            node_num, node_num
        )

    return child


def process(pop, parent, crossover_rate, mutation_rate, fitness):
    """process the population with crossover and mutation"""
    offspring = crossover(pop, parent, crossover_rate, fitness)
    offspring = mutate(offspring, mutation_rate)

    return offspring
