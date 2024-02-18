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
