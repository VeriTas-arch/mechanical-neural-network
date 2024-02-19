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


class HexaLattice:
    """Main class for HexaLattice simulation"""

    def __init__(self):
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
        self.draw_option = pymunk.pygame_util.DrawOptions(self.screen)

        # initialize the objects from the external classes
        self.beam = Beam(self)
        self.node = Node(self)
        self.operations = Operations(self)

        # initialize the lists
        self.node_list = []
        self.init_pos = []
        self.length = self.settings.length
        self.row_lenh = self.settings.row_lenh
        self.row_num = self.settings.row_num

        # create the nodes and beams
        self._create_nodes(self.space)
        self._create_beams()

        self.running = True

    def run_game(self):
        while self.running:
            self._check_events()
            self._update_screen()
            self.space.step(1/2)
            self.clock.tick(self.settings.fps)

    def _check_events(self):
        """Respond to user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
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

        # draw the arrow
        self.operations.draw_arrow(body_1.position, body_1.position + (force_1_x * 0.5, force_1_y * 0.5))
        self.operations.draw_arrow(body_2.position, body_2.position + (force_2_x * 0.5, force_2_y * 0.5))

        self.operations.draw_arrow(self.init_pos[10], self.node_list[10].position, "green")
        self.operations.draw_arrow(self.init_pos[11], self.node_list[11].position, "green")

        pygame.display.flip()

    # function that initializes the beams
    def _create_beams(self):
        length = self.length
        n = self.row_lenh
        notion_mat = np.zeros((length, length))

        for i in range(length):
            for j in range(i + 1, length):
                if i % (2 * n + 1) != n and i % (2 * n + 1) != 2 * n:
                    if j == i + n or j == i + n + 1 or j == i + 2 * n + 1:
                        notion_mat[i][j] = 1
                        self.beam.add_beam(self.node_list[i], self.node_list[j])

                elif i % (2 * n + 1) == n and j == i + n + 1:
                    notion_mat[i][j] = 1
                    self.beam.add_beam(self.node_list[i], self.node_list[j])

                elif i % (2 * n + 1) == 2*n and j == i + n:
                    notion_mat[i][j] = 1
                    self.beam.add_beam(self.node_list[i], self.node_list[j])

        # print(notion_mat)
        return notion_mat

    # function that initializes the nodes
    def _create_nodes(self, space):
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
                self.node_list.append(self.node.add_static_node(space, radius, (pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))

            else:
                self.node_list.append(self.node.add_float_node(space, radius, mass, (pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))


if __name__ == '__main__':
    MNN = HexaLattice()
    MNN.run_game()
