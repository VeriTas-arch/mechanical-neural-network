import numpy as np


class Settings:
    """class to store the settings of the stimulation environment"""

    def __init__(self):
        """initilize the settings of the stimulation environment"""
        # screen settings
        self.screen_width = 1280
        self.screen_height = 720
        self.fps = 60
        self.bg_color = (230, 230, 230)
        self.gravity = (0, 0)
        self.step = 1 / 2

        # the number of the first row
        self.row_lenh = 4
        # the number of the first column
        self.col_lenh = 2
        # the number of the rows
        self.row_num = 2 * self.col_lenh + 1
        # the length of the node list
        self.length = int((2 * self.row_lenh + 1) * self.col_lenh + self.row_lenh)

        # force settings
        self.force_1 = (0, 10)
        self.force_2 = (0, 10)
        self.force_h = (0, 10)

        # opearation settings
        self.arrow_color = (255, 0, 0)
        self.arrow_thickness = 5
        self.arrow_head_length = 10
        self.arrow_head_width = 8

        # beam settings
        self.damping = 20
        self.friction = 1
        self.beam_length = 100

        # node settings
        self.float_node_color = (0, 0, 255, 100)
        self.node_radius = 20
        self.float_node_mass = 10

        """evolution algorithm settings"""
        # connection matrix size, which is C(length, 2)
        self.pop_length = int(self.length * (self.length - 1) / 2)

        # population size for each process
        self.POP_SIZE = 500

        # process number, i.e. the core number of the CPU
        self.N_CORES = 4

        self.DNA_SIZE = self.length
        self.N_GENERATIONS = 100

        # mutation and crossover rate range
        self.CROSSOVER_RATE = np.arange(0.5, 0.8, 0.02)
        self.MUTATION_RATE = np.arange(0.02, 0.12, 0.01)

        # stability analysis settings
        self.stability_bias = 1e-2
        self.stability_inf = 30
