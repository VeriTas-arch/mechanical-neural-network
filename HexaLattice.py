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
        pygame.init()
        self.clock = pygame.time.Clock()
        self.settings = Settings()

        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        pygame.display.set_caption("HexaLattice")        
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = self.settings.gravity
        self.draw_option = pymunk.pygame_util.DrawOptions(self.screen)

        self.beam = Beam(self)
        self.node = Node(self)
        self.operations = Operations(self)

        self.node_list = []
        self.init_pos = []
        self.length = self.settings.length

        self._create_nodes(self.space)
        self._create_beams()
        
        self.running = True

    def run_game(self):
        while self.running:
            self._check_events()
            self._update_screen()
            self.space.step(1)
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
        (F1_x , F1_y) = self.settings.force_1
        (F2_x , F2_y) = self.settings.force_2
        self.operations.add_force(body_1, F1_x, F1_y)
        self.operations.add_force(body_2, F2_x, F2_y)

        # draw the arrow
        self.operations.draw_arrow(body_1.position, body_1.position + (F1_x * 0.5, F1_y * 0.5))
        self.operations.draw_arrow(body_2.position, body_2.position + (F2_x * 0.5, F2_y * 0.5))

        self.operations.draw_arrow(self.init_pos[10], self.node_list[10].position, "green")
        self.operations.draw_arrow(self.init_pos[11], self.node_list[11].position, "green")

        pygame.display.flip()

    # function that initializes the beams
    def _create_beams(self):
        length = self.length
        notion_mat = np.zeros((length, length))
        
        for i in range(length):
            for j in range (i+1, length):
                if i%5 != 2 and i%5 != 4:
                    if j == i + 2 or j == i + 3 or j == i + 5:
                        notion_mat[i][j] = 1
                        self.beam.add_beam(self.node_list[i], self.node_list[j], 10)

                elif i%5 == 2 and j == i + 3:
                        notion_mat[i][j] = 1
                        self.beam.add_beam(self.node_list[i], self.node_list[j], 10)

                elif i%5 == 4 and j == i + 2:
                        notion_mat[i][j] = 1
                        self.beam.add_beam(self.node_list[i], self.node_list[j], 10)

        return notion_mat

    # function that initializes the nodes
    def _create_nodes(self, space):
        # calculate the position bias of the nodes, note that the size of the screen is not called automatically
        sep_h = (self.settings.screen_height - self.settings.beam_length) / 2
        sep_w = (self.settings.screen_width - 4 * self.settings.beam_length * math.sin(math.pi/3)) / 2

        for i in range(0,12):
            if i%5 == 0 or i%5 == 1:
                pos_x = sep_w + 100 * math.sin(math.pi/3) + (i%5) * 2 * 100 * math.sin(math.pi/3)
                pos_y = sep_h - 100 * math.cos(math.pi/3) + 100 * (i - i%5) / 5
                self.node_list.append(self.node.add_float_node(space, 20, 10,(pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))

            elif i%5 == 3:
                pos_x = sep_w + 2 * 100 * math.sin(math.pi/3)
                pos_y = sep_h + 100 * (i - 3) / 5
                self.node_list.append(self.node.add_float_node(space, 20, 10, (pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))

            elif i%5 == 2 or i%5 == 4:
                pos_x = sep_w + (i%5 - 2) * 2 * 100 * math.sin(math.pi/3)
                pos_y = sep_h + 100 * (i - i%5) / 5
                self.node_list.append(self.node.add_static_node(space, 20, (pos_x, pos_y)))
                self.init_pos.append((pos_x, pos_y))


if __name__ == '__main__':
    MNN = HexaLattice()
    MNN.run_game()