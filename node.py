import math
import pymunk
import pymunk.pygame_util

class Node:
    """class for creating nodes in the simulation"""

    def __init__(self, me_network):
        self.settings = me_network.settings
        self.float_node_color = self.settings.float_node_color
        
    # function that adds a new float node
    def add_float_node(self, space, radius, mass, pos):
        body = pymunk.Body()
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.mass = mass
        shape.color = self.float_node_color

        space.add(body, shape)
        return body

    # function that adds a static node
    def add_static_node(self, space, radius, pos):
        body = pymunk.Body(body_type = pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Circle(body, radius)

        space.add(body, shape)
        return body