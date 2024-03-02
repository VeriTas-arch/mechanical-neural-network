import math
import pygame


class Operations:
    """class that contains all the operations that can be performed on the simulation"""

    def __init__(self, me_network):
        self.settings = me_network.settings

    def add_force(self, body, force_vector_x, force_vector_y):
        """function that adds a instantaneous force to the object"""
        body.apply_force_at_local_point((force_vector_x, force_vector_y), (0, 0))
