class Settings:
    """class to store the settings of the stimulation environment"""

    def __init__(self):
        """initilize the settings of the stimulation environment"""
        # screen settings
        self.screen_width = 1080
        self.screen_height = 720
        self.fps = 30
        self.bg_color = (230, 230, 230)
        self.gravity = (0, 0)

        # the number of the first row and the first column
        self.row_lenh = 4
        self.column_lenh = 5

        # the length of the node list
        self.length = int((2 * self.row_lenh + 1) * self.column_lenh + self.row_lenh)

        # force settings
        self.force_1 = (60, 80)
        self.force_2 = (0, 100)

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
