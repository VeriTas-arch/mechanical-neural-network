class Settings:
    """class to store the settings of the stimulation environment"""
    
    def __init__(self):
        """initilize the settings of the stimulation environment"""
        # screen settings
        self.screen_width = 1080
        self.screen_height = 720
        self.fps = 60
        self.bg_color = (230, 230, 230)
        self.gravity = (0, 0)

        # opearation settings
        self.arrow_color = (255, 0, 0)
        self.arrow_thickness = 5
        self.arrow_head_length = 10 
        self.arrow_head_width = 8

        # beam settings
        self.damping = 10

        # node settings
        self.float_node_color = (0, 0, 255, 100)