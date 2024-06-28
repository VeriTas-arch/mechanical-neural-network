import math

import numpy as np

from settings import Settings


class Eva:
    """class for Evolutionary Algorithm (EVA)"""

    def __init__(self):
        self.max_fitness = 0
        self.best_ind = []


set = Settings()
node_num = set.length
POP_SIZE = set.POP_SIZE
DNA_SIZE = set.DNA_SIZE

Array_Individual = np.ndarray[tuple[int, int], np.float64]
Array_Population = np.ndarray[tuple[int, int, int], np.float64]


def target_function():
    """set sine function as target function"""

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


def select_parent(
    pop: Array_Population,
    fitness: np.ndarray[int, np.float64],
) -> Array_Population:
    """
    Selects a parent from the population based on fitness.

    Args:
        pop (Array_Population): The population of individuals.
        fitness (np.ndarray[int, np.float64]): The fitness values of the individuals.

    Returns:
        Array_Population: The selected parent.
    """

    parent_index = np.random.choice(
        POP_SIZE, size=POP_SIZE, replace=True, p=fitness / sum(fitness)
    )

    return [pop[parent_index[i]] for i in range(POP_SIZE)]


def crossover(
    pending_pop: Array_Population,
    parent: Array_Individual,
    crossover_rate: float,
    fitness: np.ndarray[int, np.float64],
    type=0,
) -> Array_Individual:
    """
    Perform crossover operation between parent and pending population.

    Args:
        pending_pop (Array_Population): The pending population.
        parent (Array_Individual): The parent individual.
        crossover_rate (float): The probability of performing crossover.
        fitness (np.ndarray[int, np.float64]): The fitness values of the population.
        type (int, optional): The type of crossover operation. Defaults to 0.

    Returns:
        Array_Individual: The resulting individual after crossover.
    """

    if np.random.rand() < crossover_rate:
        point_num = int(DNA_SIZE / 10) + 1
        point = np.random.choice(
            np.arange(1, DNA_SIZE - 1), replace=False, size=point_num
        )
        cross_index = np.random.choice(POP_SIZE, p=fitness / sum(fitness))

        for k in range(point_num):
            if type == 0:
                pending_pop = [
                    pending_pop[cross_index][i] if abs(i - point[k]) < 2 else parent[i]
                    for i in range(node_num)
                ]
            if type == 1:
                pending_pop = [
                    np.concatenate(
                        (parent[i][: point[k]], pending_pop[cross_index][i][point[k] :])
                    )
                    for i in range(node_num)
                ]

            return pending_pop
    else:
        return parent


def mutate(child: Array_Individual, mutation_rate: float) -> Array_Individual:
    """
    Mutation operator.

    Args:
        child (Array_Individual): The individual to be mutated.
        mutation_rate (float): The probability of mutation.

    Returns:
        Array_Individual: The mutated individual.
    """

    if np.random.rand() < mutation_rate:
        # child = np.array(child)
        interval = np.max(child) - np.min(child)
        point = np.random.randint(0, DNA_SIZE)

        # mutation process
        child = interval / (1 + child) + (point + 1) * np.random.rand(
            node_num, node_num
        )

    return child


def process(
    pending_pop: Array_Population,
    parent: Array_Individual,
    crossover_rate: float,
    mutation_rate: float,
    fitness: np.ndarray[int, np.float64],
) -> Array_Individual:
    """
    Process the pending population by performing crossover and mutation operations.

    Args:
        pending_pop (Array_Population): The pending population of individuals.
        parent (Array_Individual): The parent individual used for crossover.
        crossover_rate (float): The rate at which crossover is performed.
        mutation_rate (float): The rate at which mutation is performed.
        fitness (np.ndarray[int, np.float64]): The fitness values of the individuals.

    Returns:
        Array_Individual: The offspring population after performing crossover and mutation.
    """

    offspring = crossover(pending_pop, parent, crossover_rate, fitness)
    offspring = mutate(offspring, mutation_rate)

    return offspring
