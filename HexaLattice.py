import sys
import pygame
import pymunk
import pymunk.pygame_util
import math

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

        self.node._create_nodes()
        self.beam._create_beams()
        
        self.running = True

    def run_game(self):
        while self.running:
            self._check_events()
            self._update_screen()
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

        pygame.display.flip()


if __name__ == '__main__':
    MNN = HexaLattice()
    MNN.run_game()