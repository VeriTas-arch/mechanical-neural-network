import sys
import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np

from settings import Settings
from beam import Beam
from node import Node
from operations import Operations
from eva import Eva
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

        # initialize pymunk
        self.space = pymunk.Space()
        self.space.gravity = self.settings.gravity
        # self.draw_option = pymunk.pygame_util.DrawOptions(self.screen)

        # initialize the objects from the external classes
        self.beam = Beam(self)
        self.node = Node(self)
        self.operations = Operations(self)

        # initialize the lists
        self.node_list = []
        self.init_pos = []
        self.dynamic_pos = []

        self.length = self.settings.length
        self.stiffness_mat = stiffness_mat
        self.row_lenh = self.settings.row_lenh
        self.row_num = self.settings.row_num

        # create the nodes and beams
        self._create_nodes(self.space)
        self._create_beams()

        self.running = True

    def run_game(self):
        while self.running:
            # self._check_events()
            self._update_screen()
            self._check_stability()
            self.space.step(2)
            self.clock.tick(self.settings.fps)

    def _check_events(self):
        """Respond to user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                sys.exit()

    def _update_screen(self):
        """Update the screen"""
        # self.screen.fill(self.settings.bg_color)
        # self.space.debug_draw(self.draw_option)

        # record the dynamic position of the nodes
        self._update_pos()

        # apply force to the nodes
        body_1 = self.node_list[0]
        body_2 = self.node_list[1]
        (force_1_x, force_1_y) = self.settings.force_1
        (force_2_x, force_2_y) = self.settings.force_2
        self.operations.add_force(body_1, force_1_x, force_1_y)
        self.operations.add_force(body_2, force_2_x, force_2_y)

        # pygame.display.flip()

    def _create_beams(self):
        """function that initializes the beams"""
        length = self.length
        n = self.row_lenh
        # notion_mat = np.zeros((length, length))
        stiffness_mat = self.stiffness_mat

        for i in range(length):
            for j in range(i + 1, length):
                if i % (2 * n + 1) != n and i % (2 * n + 1) != 2 * n:
                    if j == i + n or j == i + n + 1 or j == i + 2 * n + 1:
                        # notion_mat[i][j] = 1
                        self.beam.add_beam(self.node_list[i], self.node_list[j], stiffness_mat[i][j])

                elif i % (2 * n + 1) == n and j == i + n + 1:
                    # notion_mat[i][j] = 1
                    self.beam.add_beam(self.node_list[i], self.node_list[j], stiffness_mat[i][j])

                elif i % (2 * n + 1) == 2*n and j == i + n:
                    # notion_mat[i][j] = 1
                    self.beam.add_beam(self.node_list[i], self.node_list[j], stiffness_mat[i][j])

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
            self.dynamic_pos.append((0, 0))
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
                self.node_list.append(self.node.add_static_node(space, radius, (pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))

            else:
                self.node_list.append(self.node.add_float_node(space, radius, mass, (pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))

    def _check_stability(self):
        """check if the network is stable"""
        bias = 0
        for i in range(len(self.node_list)):
            x_bias = (self.node_list[i].position[0] - self.dynamic_pos[i][0]) ** 2
            y_bias = (self.node_list[i].position[1] - self.dynamic_pos[i][1]) ** 2
            bias += math.sqrt(x_bias + y_bias)

        if bias < self.settings.stability_bias:
            self.running = False
            return True

    def _update_pos(self):
        """record the dynamic position of the nodes"""
        for i in range(len(self.node_list)):
            self.dynamic_pos[i] = self.node_list[i].position


if __name__ == '__main__':
    """define the EVA functions and initialize the parameters"""

    set = Settings()
    eva = Eva()

    node_num = set.length

    def fillBits(size):
        return 1 << size - 1

    POP_SIZE = set.POP_SIZE
    N_GENERATIONS = set.N_GENERATIONS
    DNA_SIZE = set.DNA_SIZE
    MUTATION_RATE = set.MUTATION_RATE

    def target_function():
        """set sine function as target function"""
        T = set.screen_width
        omiga = 2 * math.pi / T
        Amp = set.beam_length / 2
        bias = Amp / 2
        return lambda x: Amp * math.sin(omiga * x) + set.screen_height + bias

    def avoid_function():
        """set the function that the input nodes should avoid"""
        T = set.screen_width
        omiga = 2 * math.pi / T
        Amp = set.beam_length / 2
        bias = 0
        return lambda x: Amp * math.sin(omiga * x + math.pi) + bias

    def get_fitness(indPos):
        """calculate the fitness of a certain individual"""
        target = target_function()
        avoid = avoid_function()
        length = set.row_lenh
        num = node_num
        sum_input = 0
        sum_output = 0

        for i in range(length):
            bias_in = (avoid(indPos[i][0]) - indPos[i][1]) ** 2
            bias_out = (target(indPos[num - i - 1][0]) - indPos[num - i - 1][1]) ** 2
            sum_input += bias_in
            sum_output += bias_out

        rms_in = math.sqrt(sum_input / length)
        rms_out = math.sqrt(sum_output / length)
        fitness = rms_in ** (1/2) + 1/(rms_out ** 2)

        return fitness

    def select_parent(pop, fitness):
        """choose the parent based on fitness"""
        fitness = np.array(fitness)
        index = np.random.choice(POP_SIZE, size=POP_SIZE, replace=True, p=fitness / sum(fitness))
        temp = []
        for i in range(POP_SIZE):
            temp.append(pop[index[i]])
        return temp

    def crossover(pop, parent):
        """crossover the parents to generate offspring"""
        # choose an individual from the population to crossover
        index = np.random.randint(0, POP_SIZE - 1)
        # choose a crossover point
        point = np.random.randint(1, DNA_SIZE - 1)

        crossover_result = []

        for i in range(node_num):
            crossover_result.append([])
            crossover_result[i].extend(parent[i][:point])
            crossover_result[i].extend(pop[index][i][point:])
            np.random.shuffle(crossover_result[i])

        return crossover_result

    def mutate(child):
        """mutation operator"""
        for i in range(DNA_SIZE):
            if np.random.rand() < MUTATION_RATE:
                point = 1 << i
                for j in range(node_num):
                    for k in range(node_num):
                        child[j][k] = point / (1 + child[j][k]) + point * np.random.rand() / 3

        return child

    pop = []
    for i in range(POP_SIZE):
        # generate a random stiffness matrix as the initial population
        stiffness_mat = np.random.rand(node_num, node_num) * 20
        pop.append(stiffness_mat)

    for gen in tqdm(range(N_GENERATIONS), colour='red', desc='EVA'):
        """initialize the population and the population's position"""
        # the population that is to be evolved, whose individuals are HexaLattice objects
        popGame = []
        # store the positions of nodes in each individual
        pop_pos = []
        # store the fitness of each individual
        fitness = []

        for i in range(POP_SIZE):
            popGame.append(HexaLattice(pop[i]))
            fitness.append(0)
            pop_pos.append([])
            for j in range(node_num):
                pop_pos[i].append(popGame[i].node_list[j].position)

        """evolution process"""
        # calculate the fitness of the current generation
        for i in range(POP_SIZE):
            popGame[i].run_game()
            for j in range(node_num):
                pop_pos[i][j] = popGame[i].node_list[j].position

            # get the fitness of the population
            fitness[i] = get_fitness(pop_pos[i])

            # record the best individual
            if fitness[i] > eva.max_fitness:
                eva.max_fitness = fitness[i]
                eva.best_ind = pop[i].copy()

        # chosse the parent based on fitness
        pop = select_parent(pop, fitness)
        popCopy = pop.copy()

        for popIndex in range(POP_SIZE):
            parent = pop[popIndex]
            # single-point crossover
            child = crossover(popCopy, parent)
            child = mutate(child)

            # survival selection, use age-based replacement
            # the number of parents and children is the same, so all children are replaced with parents
            pop[popIndex] = child

        sleep(0.01)

        if gen % 200 == 0:
            np.savetxt('temp.csv', eva.best_ind, delimiter=',')

    sleep(0.01)

    # print the best individual
    np.savetxt('individual.csv', eva.best_ind, delimiter=',')

    print("Best individual: ", eva.best_ind)
    print("Fitness: ", eva.max_fitness)
