import pymunk
import pymunk.pygame_util
import math
import EVA
import time
import plot

import numpy as np

from settings import Settings
from beam import Beam
from node import Node
from operations import Operations
from multiprocessing import Process, Queue


class HexaLattice:
    """Main class for HexaLattice simulation"""

    def __init__(self, stiffness_mat):
        """program initialization"""
        self.settings = Settings()

        # initialize pymunk
        self.space = pymunk.Space()
        self.space.gravity = self.settings.gravity

        """initialize external classes"""
        self.beam = Beam(self)
        self.node = Node(self)
        self.operations = Operations(self)

        """define the constants"""
        # simulation parameters
        self.length = self.settings.length
        self.step = self.settings.step
        self.row_lenh = self.settings.row_lenh
        self.col_lenh = self.settings.col_lenh
        self.row_num = self.settings.row_num

        # node and beam parameters
        self.blen = self.settings.beam_length
        self.radius = self.settings.node_radius
        self.mass = self.settings.float_node_mass
        self.fnum = self.length - self.col_lenh * 2

        # execution parameters
        self.stiffness_mat = stiffness_mat
        # self.step_counter = 0

        # stability parameters
        self.max_v_1 = 0
        self.max_v_2 = 0

        """initialize the lists"""
        self.fnode_index = np.zeros(self.length)
        self.beam_index = np.zeros((self.length, self.length))

        # self.dynamic_pos = [None for i in range(self.length)]
        self.init_pos = [None for i in range(self.length)]
        self.node_record = [None for i in range(self.length)]
        self.node_list = [None for i in range(self.length)]
        self.beam_list = [
            [None for j in range(self.length)] for i in range(self.length)
        ]

        # create the nodes and beams
        self._init_pos()
        self._init_index()
        self._create_nodes(self.space)
        self._create_beams(self.stiffness_mat)

        self.running = True

    def run_game(self):
        """Main game loop"""
        for _ in range(300):
            self._update_screen()
            # self._check_stability(self.settings.stablity_bias)
            # self.step_counter += 1
            self.space.step(self.step)

    def _update_screen(self):
        """Update the screen with force function"""
        # apply force to the nodes
        body = [self.node_list[i] for i in range(self.row_lenh)]
        (force_x, force_y) = self.settings.force_h

        for i in range(self.row_lenh):
            self.operations.add_force(body[i], force_x, force_y)

        # apply frictions
        for i in range(self.settings.length):
            v = math.sqrt(
                self.node_list[i].velocity[0] ** 2 + self.node_list[i].velocity[1] ** 2
            )
            f = self.settings.friction
            F = math.sqrt(
                self.node_list[i].force[0] ** 2 + self.node_list[i].force[1] ** 2
            )

            if v >= 1e-2:
                e = self.node_list[i].velocity / v
                friction_x, friction_y = -f * e[0], -f * e[1]
            elif v < 1e-2 and F < f:
                friction_x, friction_y = (
                    -self.node_list[i].force[0],
                    -self.node_list[i].force[1],
                )
            else:
                e = self.node_list[i].force / F
                friction_x, friction_y = -f * e[0], -f * e[1]

            self.operations.add_force(self.node_list[i], friction_x, friction_y)

    def _init_pos(self):
        """calculate the initial position of the nodes"""
        sep_x = (
            self.settings.screen_width - (self.row_lenh - 1) * self.blen * math.sqrt(3)
        ) / 2
        sep_y = (self.settings.screen_height - ((self.row_num - 1) / 2) * self.blen) / 2

        n = self.row_lenh
        T = 2 * self.row_lenh + 1
        column_counter = 0
        row_counter = 0

        for i in range(self.length):
            if column_counter == n or column_counter == T:
                column_counter = column_counter % T
                row_counter += 1

            if column_counter < n:
                pos_x = sep_x + column_counter * self.blen * math.sqrt(3)
                pos_y = sep_y + row_counter * self.blen / 2
            else:
                pos_x = (
                    sep_x
                    + (column_counter - n) * self.blen * math.sqrt(3)
                    - self.blen * math.sqrt(3) / 2
                )
                pos_y = sep_y + row_counter * self.blen / 2

            column_counter += 1
            self.init_pos[i] = (pos_x, pos_y)

    def _init_index(self):
        """calculate the index of the nodes and beams"""
        n = self.row_lenh
        T = 2 * self.row_lenh + 1

        # calculate the index of the float nodes
        self.fnode_index = [
            False if (i + 1) % T == 0 or (i + 1) % T == n + 1 else True
            for i in range(self.length)
        ]

        # calculate the index of the beams
        for i in range(self.length):
            for j in range(i + 1, self.length):
                if i % T != n and i % T != 2 * n:
                    if j == i + n or j == i + n + 1 or j == i + 2 * n + 1:
                        self.beam_index[i][j] = True

                elif i % T == n and j == i + n + 1:
                    self.beam_index[i][j] = True

                elif i % T == 2 * n and j == i + n:
                    self.beam_index[i][j] = True

    def _create_nodes(self, space):
        """function that initializes the nodes"""
        self.node_record = [
            (
                self.node.add_float_node(
                    space, self.radius, self.mass, self.init_pos[i]
                )
                if self.fnode_index[i]
                else self.node.add_static_node(space, self.radius, self.init_pos[i])
            )
            for i in range(self.length)
        ]
        self.node_list = [node[0] for node in self.node_record]

    def _create_beams(self, stiffness_mat):
        """function that initializes the beams"""
        temp = [
            (
                self.beam.add_beam(
                    self.node_list[i], self.node_list[j], stiffness_mat[i][j]
                )
                if self.beam_index[i][j]
                else None
            )
            for i in range(self.length)
            for j in range(self.length)
        ]

        len_temp = len(temp)
        self.beam_list = [
            temp[i : i + self.length] for i in range(0, len_temp, self.length)
        ]

    def _create_float_nodes(self, space):
        """function that initializes the float nodes"""
        temp = self.node_record.copy()
        self.node_record = [
            (
                self.node.add_float_node(
                    space, self.radius, self.mass, self.init_pos[i]
                )
                if bool(self.fnode_index[i])
                else temp[i]
            )
            for i in range(self.length)
        ]
        self.node_list = [node[0] for node in self.node_record]

    def _delete_float_nodes(self):
        """function that removes the float nodes"""
        for i in range(self.length):
            if bool(self.fnode_index[i]):
                self.space.remove(self.node_record[i][0])
                self.space.remove(self.node_record[i][1])
                self.node_list[i] = None

    def _delete_beams(self):
        """function that removes the beams"""
        for i in range(self.length):
            for j in range(i + 1, self.length):
                if self.beam_list[i][j] is not None:
                    self.space.remove(self.beam_list[i][j])
                    self.beam_list[i][j] = None

    def _check_stability(self, threshold):
        v_1 = (
            self.node_list[set.length - 2].velocity[0] ** 2
            + self.node_list[set.length - 2].velocity[1] ** 2
        )
        v_2 = (
            self.node_list[set.length - 1].velocity[0] ** 2
            + self.node_list[set.length - 1].velocity[1] ** 2
        )

        self.max_v_1 = max(v_1, self.max_v_1)
        self.max_v_2 = max(v_2, self.max_v_2)

        if self.max_v_1 > 0.3 or self.max_v_2 > 0.3:
            if v_1 <= threshold and v_2 <= threshold:
                self.running = False
                self.max_v_1 = 0
                self.max_v_2 = 0

    def _reset_game(self, stiffness_mat):
        """reset the game"""
        self._delete_float_nodes()
        self._delete_beams()
        self._create_float_nodes(self.space)
        self._create_beams(stiffness_mat)

        self.running = True


set = Settings()


def pymunk_run(queue, process_num, popGame, rate):
    """function that runs the pymunk simulation"""

    # initialize the parameters
    max_fitness = 0
    best_ind = None
    node_num = set.length
    N_GENERATIONS = set.N_GENERATIONS
    POP_SIZE = set.POP_SIZE

    # initialize the population and the population's position
    pop = np.random.rand(POP_SIZE, node_num, node_num) * 25
    pop_pos = np.zeros((POP_SIZE, node_num, 2))
    fitness = np.zeros(POP_SIZE)
    fit_data = np.zeros((N_GENERATIONS, 2))

    # detect_interval = int(N_GENERATIONS / 100)

    for gen in range(N_GENERATIONS):

        for i in range(POP_SIZE):
            popGame._reset_game(pop[i])
            popGame.run_game()

            pop_pos[i] = [popGame.node_list[j].position for j in range(node_num)]
            # print(pop_pos[index])

        # get the fitness of the population
        fitness = np.array(
            [ind_fitness for ind_fitness in map(EVA.get_fitness, pop_pos)]
        )

        fit_data[gen] = [max(fitness), np.mean(fitness)]

        # sort the population based on fitness
        sort_fitness = np.argsort(fitness)
        pop_fitness = np.array([pop[i] for i in sort_fitness])

        # record the best individual
        index = sort_fitness[POP_SIZE - 1]
        if fitness[index] > max_fitness:
            max_fitness = fitness[index]
            best_ind = pop[index]

            if process_num == 0:
                print(f"\nthe current best fitness is {max_fitness}")

        # chosse the parent based on fitness
        pop = EVA.select_parent(pop, fitness)
        popCopy = pop.copy()
        pop = [
            EVA.process(
                popCopy,
                pop[popIndex],
                rate[process_num][0],
                rate[process_num][1],
                fitness,
            )
            for popIndex in range(POP_SIZE)
        ]

        # create the new population
        fit_point = np.random.choice(POP_SIZE, p=sort_fitness / sum(sort_fitness))
        pop = np.concatenate(
            (pop[:fit_point], pop_fitness[fit_point:]), axis=None
        ).reshape(POP_SIZE, node_num, node_num)

        if process_num == 0:
            progress = gen / N_GENERATIONS * 100
            print(f"\nEVA{process_num} is processing {progress:.2f}%")

    # save the result
    result = (max_fitness, best_ind)
    np.savetxt(
        f"./storage/multiprocessing/individual{process_num}.csv",
        result[1],
        delimiter=",",
    )
    np.savetxt(
        f"./storage/multiprocessing/fitness_data/fitness{process_num}.csv",
        fit_data,
        delimiter=",",
    )
    print(f"\nthe best fitness of EVA{process_num} is {result[0]}")
    print(
        f"crossover rate{process_num}: {rate[process_num][0]}, mutation rate{process_num}: {rate[process_num][1]}"
    )
    # queue.put(result)


if __name__ == "__main__":
    """initialize the population and the population's position"""
    POP_SIZE = set.POP_SIZE
    node_num = set.length
    process_num = set.N_CORES

    init_stiffness = np.random.rand(node_num, node_num) * 20

    # randomly initialize the possibilities
    rate = [
        (np.random.choice(set.CROSSOVER_RATE), np.random.choice(set.MUTATION_RATE))
        for _ in range(process_num)
    ]

    popGame = HexaLattice(init_stiffness)
    print("\nEvolution starts")

    queue = Queue()

    st = time.time()

    """evolution process"""
    process_list = []

    for i in range(process_num):
        p = Process(target=pymunk_run, args=(queue, i, popGame, rate))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    ed = time.time()
    print(f"\nEvolution ends, total time: {ed - st:.2f}s")

    plot.plot_fitness(rate)

    # for i in range(process_num):
    # result = queue.get()
