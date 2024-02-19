import pymunk
import pymunk.pygame_util


class Beam:
    """class for creating beams in the simulation"""
    def __init__(self, me_network):
        self.space = me_network.space
        self.settings = me_network.settings
        self.damping = self.settings.damping
        self.beam_length = self.settings.beam_length

    # fuction that adds a spring (i.e. beam)
    def add_beam(self, body_1, body_2, stiffness=10):
        damping = self.damping
        beam_length = self.beam_length
        spring = pymunk.DampedSpring(body_1, body_2, (0, 0), (0, 0), beam_length, stiffness, damping)
        self.space.add(spring)
