class Settings:
    """class to store the settings of the stimulation environment"""

    def __init__(self):
        """initilize the settings of the stimulation environment"""
        # screen settings
        self.screen_width = 720
        self.screen_height = 540
        self.fps = 240
        self.bg_color = (230, 230, 230)
        self.gravity = (0, 0)
        self.step = 1

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
        self.damping = 20
        self.beam_length = 100

        # node settings
        self.float_node_color = (0, 0, 255, 100)
        self.node_radius = 20
        self.float_node_mass = 10

        """evolution algorithm settings"""
        # population size, which is C(length, 2)
        self.pop_length = int(self.length * (self.length - 1) / 2)
        self.scale_factor = 10

        self.POP_SIZE = 50
        self.DNA_SIZE = self.length
        self.N_GENERATIONS = 10
        self.MUTATION_RATE = 0.05

        # stability analysis settings
        self.stability_bias = 1e-30
        self.stability_inf = 30

        # resume and record settings
        self.resume = False
        self.record = True
        self.record_interval = 100
