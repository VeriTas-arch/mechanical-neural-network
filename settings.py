class Settings:
    """class to store the settings of the stimulation environment"""

    def __init__(self):
        """initilize the settings of the stimulation environment"""
        # screen settings
        self.screen_width = 960
        self.screen_height = 540
        self.fps = 60
        self.bg_color = (230, 230, 230)
        self.gravity = (0, 0)

        # the number of the first row
        self.row_lenh = 2
        # row_num should be an odd number
        self.row_num = 5
        # the length of the node list
        self.length = int((2 * self.row_lenh + 1) * (self.row_num - 1)/2 + self.row_lenh)

        # force settings
        self.force_1 = (0, 50)
        self.force_2 = (0, 50)

        # opearation settings
        self.arrow_color = (255, 0, 0)
        self.arrow_thickness = 5
        self.arrow_head_length = 10
        self.arrow_head_width = 8

        # beam settings
        self.damping = 10
        self.beam_length = 100

        # node settings
        self.float_node_color = (0, 0, 255, 100)
        self.node_radius = 20
        self.float_node_mass = 10

        # evolution algorithm settings
        # population size, which is C(length, 2)
        self.pop_length = int(self.length * (self.length - 1) / 2)
        self.scale_factor = 10

        self.POP_SIZE = 100
        self.DNA_SIZE = self.length
        self.N_GENERATIONS = 100
        self.MUTATION_RATE = 0.01
        self.stability_bias = 0.001
