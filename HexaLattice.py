import sys
import pygame
import pymunk
import pymunk.pygame_util
import math
import EVA

import numpy as np
import matplotlib.pyplot as plt

from settings import Settings
from beam import Beam
from node import Node
from operations import Operations
from time import sleep
from tqdm import tqdm


class HexaLattice:
    """Main class for HexaLattice simulation"""

    def __init__(self, stiffness):
        # initialize pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.settings = Settings()
        self.screen = pygame.display.set_mode(
            (self.settings.screen_width, self.settings.screen_height))
        pygame.display.set_caption("Mechanical Neural Network")
        self.step_counter = 0
        self.step_interval = 50

        # initialize pymunk
        self.space = pymunk.Space()
        self.space.gravity = self.settings.gravity
        self.draw_option = pymunk.pygame_util.DrawOptions(self.screen)

        # initialize the objects from the external classes
        self.beam = Beam(self)
        self.node = Node(self)
        self.operations = Operations(self)

        # initialize the lists
        self.node_list = []
        self.beam_list = []
        self.init_pos = []
        self.dynamic_pos = []
        self.node_record = []

        for i in range(self.settings.length):
            self.node_list.append(None)
            self.node_record.append(None)
            self.init_pos.append((0, 0))
            self.dynamic_pos.append((0, 0))
            self.beam_list.append([])

            for j in range(self.settings.length):
                self.beam_list[i].append(None)

        self.length = self.settings.length
        self.stiffness_mat = stiffness_mat
        self.row_lenh = self.settings.row_lenh
        self.row_num = self.settings.row_num

        # create the nodes and beams
        self._create_nodes(self.space)
        self._create_beams(self.stiffness_mat)

        self.running = True

    def run_game(self):
        """Main game loop"""
        # while self.running:
        for _ in range(self.step_interval):
            self._check_events()
            self._update_screen()
            self._check_stability()
            self.step_counter += 1
            self.space.step(1)
            self.clock.tick(self.settings.fps)

            # record the dynamic position of the nodes
            self._update_pos()

    def _check_events(self):
        """Respond to user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    def _update_screen(self):
        """Update the screen"""
        self.screen.fill(self.settings.bg_color)
        self.space.debug_draw(self.draw_option)

        # apply force to the nodes
        body_1 = self.node_list[0]
        body_2 = self.node_list[1]
        (force_1_x, force_1_y) = self.settings.force_1
        (force_2_x, force_2_y) = self.settings.force_2
        self.operations.add_force(body_1, force_1_x, force_1_y)
        self.operations.add_force(body_2, force_2_x, force_2_y)

        pygame.display.flip()

    def _create_beams(self, stiffness_mat):
        """function that initializes the beams"""
        length = self.length
        n = self.row_lenh
        # notion_mat = np.zeros((length, length))

        for i in range(length):
            for j in range(i + 1, length):
                if i % (2 * n + 1) != n and i % (2 * n + 1) != 2 * n:
                    if j == i + n or j == i + n + 1 or j == i + 2 * n + 1:
                        # notion_mat[i][j] = 1
                        self.beam_list[i][j] = self.beam.add_beam(
                            self.node_list[i], self.node_list[j], stiffness_mat[i][j])

                elif i % (2 * n + 1) == n and j == i + n + 1:
                    # notion_mat[i][j] = 1
                    self.beam_list[i][j] = self.beam.add_beam(
                        self.node_list[i], self.node_list[j], stiffness_mat[i][j])

                elif i % (2 * n + 1) == 2*n and j == i + n:
                    # notion_mat[i][j] = 1
                    self.beam_list[i][j] = self.beam.add_beam(
                        self.node_list[i], self.node_list[j], stiffness_mat[i][j])

        # print(notion_mat)
        # return notion_mat

    def _create_nodes(self, space):
        """function that initializes the nodes"""
        blen = self.settings.beam_length
        radius = self.settings.node_radius
        mass = self.settings.float_node_mass

        sep_x = (self.settings.screen_width - (self.row_lenh - 1) * blen * math.sqrt(3))/2
        sep_y = (self.settings.screen_height - ((self.row_num - 1)/2) * blen)/2

        n = self.row_lenh
        T = 2 * n + 1
        column_counter = 0
        row_counter = 0

        for i in range(0, self.length):
            if column_counter == n:
                row_counter += 1

            if column_counter == T:
                column_counter = 0
                row_counter += 1

            if column_counter < n:
                pos_x = sep_x + column_counter * blen * math.sqrt(3)
                pos_y = sep_y + row_counter * blen / 2

            if column_counter >= n and column_counter < T:
                pos_x = sep_x + (column_counter - n) * blen * math.sqrt(3) - blen * math.sqrt(3) / 2
                pos_y = sep_y + row_counter * blen / 2
            column_counter += 1

            if (i + 1) % T == 0 or (i + 1) % T == self.row_lenh + 1:
                self.node_record[i] = self.node.add_static_node(space, radius, (pos_x, pos_y))
                self.node_list[i] = self.node_record[i][0]

            else:
                self.node_record[i] = self.node.add_float_node(space, radius, mass, (pos_x, pos_y))
                self.node_list[i] = self.node_record[i][0]

            self.init_pos[i] = (pos_x, pos_y)

    def _create_float_nodes(self, space):
        """function that initializes the float nodes"""
        n = self.row_lenh
        T = 2 * n + 1
        radius = self.settings.node_radius
        mass = self.settings.float_node_mass

        for i in range(self.length):
            if (i + 1) % T != 0 and (i + 1) % T != self.row_lenh + 1:
                self.node_record[i] = self.node.add_float_node(space, radius, mass, self.init_pos[i])
                self.node_list[i] = self.node_record[i][0]
                self.dynamic_pos[i] = self.init_pos[i]

    def _delete_float_nodes(self):
        """function that removes the float nodes"""
        for i in range(self.length):
            if self.node_list[i] is not None and self.node_list[i].body_type == pymunk.Body.DYNAMIC:
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

    def _check_stability(self):
        """check if the network is stable"""
        bias = 0
        for i in range(self.length):
            x_bias = (self.node_list[i].position[0] - self.dynamic_pos[i][0]) ** 2
            y_bias = (self.node_list[i].position[1] - self.dynamic_pos[i][1]) ** 2
            bias += x_bias + y_bias

        rms = math.sqrt(bias / self.length)

        if rms < self.settings.stability_bias and self.step_counter > self.settings.stability_inf:
            self.running = False
        return True

    def _update_pos(self):
        """record the dynamic position of the nodes"""
        for i in range(self.length):
            self.dynamic_pos[i] = self.node_list[i].position

    def _reset_game(self, stiffness_mat):
        """reset the game"""
        self._delete_float_nodes()
        self._delete_beams()
        self._create_float_nodes(self.space)
        self._create_beams(stiffness_mat)

        self.step_counter = 0
        self.running = True


if __name__ == '__main__':
    """define the EVA functions and initialize the parameters"""

    set = Settings()
    eva = EVA.Eva()

    node_num = set.length
    record_interval = set.record_interval
    POP_SIZE = set.POP_SIZE
    N_GENERATIONS = set.N_GENERATIONS
    max_fitness = eva.max_fitness
    best_ind = eva.best_ind

    """initialize the population and the population's position"""
    pop = []
    new_pop = []
    # store the positions of nodes in each individual
    pop_pos = []
    # store the fitness of each individual
    fitness = []

    for i in range(POP_SIZE):
        # generate a random stiffness matrix as the initial population
        stiffness_mat = np.random.rand(node_num, node_num) * 20
        pop.append(stiffness_mat)

        fitness.append(0)
        pop_pos.append([])
        for j in range(node_num):
            pop_pos[i].append(None)

    """RESUME from the last generation"""
    resume = set.resume
    if resume:
        stiffness_data = np.load(file="data.npy")
        pop = stiffness_data

    init_stiffness = np.random.rand(node_num, node_num) * 20
    # the population that is to be evolved, whose individuals are HexaLattice objects
    popGame = HexaLattice(init_stiffness)
    popGame.run_game()
    print("\nEvolution starts")

    """initialize the plot"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = []
    y = []

    plt.ion()

    """evolution process"""
    for gen in tqdm(range(N_GENERATIONS), colour='red', desc='EVA', dynamic_ncols=True):

        # calculate the fitness of the current generation
        for i in range(POP_SIZE):
            # pop[i] = np.random.rand(node_num, node_num) * 40 * np.random.rand()
            popGame._reset_game(pop[i])
            popGame.run_game()

            for j in range(node_num):
                pop_pos[i][j] = popGame.node_list[j].position

            # get the fitness of the population
            fitness[i] = EVA.get_fitness(pop_pos[i], node_num)

            # draw the fitness dots
            x.append(gen)
            y.append(fitness[i])

            ax.clear()
            ax.scatter(x, y, c='r', s=10)

            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('EVA')

            plt.draw()
            plt.pause(0.01)

            # record the best individual
            if fitness[i] > max_fitness:
                max_fitness = fitness[i]
                best_ind = pop[i].copy()
                print(f"\nthe current best fitness is {max_fitness}")

            # sort the population based on fitness
            sort_fitness = np.argsort(fitness)
            pop_fitness = [pop[i] for i in sort_fitness]

        # chosse the parent based on fitness
        popCopy = pop.copy()
        pop = EVA.select_parent(pop, fitness)


        for popIndex in range(POP_SIZE):
            parent = pop[popIndex]
            # single-point crossover
            child = EVA.crossover(popCopy, parent)
            child = EVA.mutate(child)

            # survival selection, use age-based replacement
            # the number of parents and children is the same, so all children are replaced with parents
            pop[popIndex] = child

        # create the new population
        fit_point = np.random.choice(POP_SIZE, p=sort_fitness / sum(sort_fitness))
        new_pop = pop[:fit_point]
        new_pop.extend(pop_fitness[fit_point:])
        pop = new_pop.copy()

        sleep(0.01)

        """RECORD the last generation"""
        record = set.record
        if record and gen % record_interval == 0:
            pop_data = pop.copy()
            temp_data = np.array(pop_data)
            np.save(file="data.npy", arr=temp_data)

    # print the best individual
    np.savetxt('individual.csv', best_ind, delimiter=',')

    print("Best individual: ", best_ind)
    print("Fitness: ", max_fitness)

    plt.ioff()
    plt.show()
