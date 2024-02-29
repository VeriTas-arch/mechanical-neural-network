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
from pathlib import Path

import EVA


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
        self.node_record = []

        for i in range(self.settings.length):
            self.node_list.append(None)
            self.node_record.append(None)
            self.init_pos.append((0, 0))
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
        for _ in range(50):
            self._check_events()
            self._update_screen()
            if self.step_counter == 200:
                print(self.node_list[7].position)
                print(self.node_list[10].position)
                print(self.node_list[11].position)
            self.step_counter += 1
            self.space.step(1)
            self.clock.tick(self.settings.fps)

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
        notion_mat = np.zeros((length, length))

        for i in range(length):
            for j in range(i + 1, length):
                if i % (2 * n + 1) != n and i % (2 * n + 1) != 2 * n:
                    if j == i + n or j == i + n + 1 or j == i + 2 * n + 1:
                        notion_mat[i][j] = 1
                        self.beam_list[i][j] = self.beam.add_beam(
                            self.node_list[i], self.node_list[j], stiffness_mat[i][j])

                elif i % (2 * n + 1) == n and j == i + n + 1:
                    notion_mat[i][j] = 1
                    self.beam_list[i][j] = self.beam.add_beam(
                        self.node_list[i], self.node_list[j], stiffness_mat[i][j])

                elif i % (2 * n + 1) == 2*n and j == i + n:
                    notion_mat[i][j] = 1
                    self.beam_list[i][j] = self.beam.add_beam(
                        self.node_list[i], self.node_list[j], stiffness_mat[i][j])

        # print(notion_mat)
        return notion_mat

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


if __name__ == '__main__':
    # read the stiffness matrix from the csv file
    path = Path(__file__).parent / "individual.csv"
    stiffness_mat = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)
    # print(stiffness_mat)

    # create the HexaLattice object
    hexalattice = HexaLattice(stiffness_mat)
    # run the game
    hexalattice.run_game()

    # calculate the fitness
    set = Settings()
    node_num = set.length
    pop_pos = []
    for j in range(node_num):
        pop_pos.append(hexalattice.node_list[j].position)
        print(hexalattice.node_list[j].position)
    print("the fitness is:", EVA.get_fitness(pop_pos, node_num))
