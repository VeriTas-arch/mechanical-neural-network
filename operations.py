import math
import pygame

class Operations:
    """class that contains all the operations that can be performed on the simulation"""
    
    def __init__(self, me_network):
        self.settings = me_network.settings
        self.screen = me_network.screen
        self.arrow_color = self.settings.arrow_color
        self.arrow_thickness = self.settings.arrow_thickness
        self.arrow_head_length = self.settings.arrow_head_length
        self.arrow_head_width = self.settings.arrow_head_width

    # function that adds a instantaneous force to the object
    def add_force(self, body, force_vector_x, force_vector_y):
        body.apply_force_at_local_point((force_vector_x, force_vector_y), (0, 0))

    # function that draws an arraw, finished by ChatGPT
    def draw_arrow(self, start_pos, end_pos, color = (255, 0, 0), thickness = 5):
        # calculate the angle and length of the arrow
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

        # create the arrow line
        pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)

        # calculate the coordinates of the arrow head
        arrow_head_x = end_pos[0] - self.arrow_head_length * math.cos(angle)
        arrow_head_y = end_pos[1] - self.arrow_head_length * math.sin(angle)

        # calculate the coordinates of the arrow head's left and right points
        arrow_head_left_x = arrow_head_x + self.arrow_head_width * math.cos(angle + math.pi / 2)
        arrow_head_left_y = arrow_head_y + self.arrow_head_width * math.sin(angle + math.pi / 2)
        
        arrow_head_right_x = arrow_head_x + self.arrow_head_width * math.cos(angle - math.pi / 2)
        arrow_head_right_y = arrow_head_y + self.arrow_head_width * math.sin(angle - math.pi / 2)

        # draw the arrow head
        pygame.draw.polygon(self.screen, color, (end_pos, (arrow_head_left_x, arrow_head_left_y), (arrow_head_right_x, arrow_head_right_y)))