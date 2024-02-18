class Operations:
    """class that contains all the operations that can be performed on the simulation"""
    
    def __init__(self):
        self.arrow_color = (255, 0, 0)
        self.arrow_thickness = 5
        self.arrow_head_length = 10 
        self.arrow_head_width = 8

    # function that adds a instantaneous force to the object
    def add_force(self, force_vector):
        body.apply_force_at_local_point(force_vector, (0, 0))

        
    # function that draws an arraw, finished by ChatGPT
    def draw_arrow(self, start_pos, end_pos, color = self.arrow_color, thickness = self.arrow_thickness):
        # calculate the angle and length of the arrow
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        length = ((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2) ** 0.5

        # create the arrow line
        pygame.draw.line(screen, color, start_pos, end_pos, thickness)

        # calculate the coordinates of the arrow head
        arrow_head_x = end_pos[0] - arrow_hear_length * math.cos(angle)
        arrow_head_y = end_pos[1] - arrow_hear_length * math.sin(angle)

        # calculate the coordinates of the arrow head's left and right points
        arrow_head_left_x = arrow_head_x + arrow_hear_width * math.cos(angle + math.pi / 2)
        arrow_head_left_y = arrow_head_y + arrow_hear_width * math.sin(angle + math.pi / 2)
        
        arrow_head_right_x = arrow_head_x + arrow_hear_width * math.cos(angle - math.pi / 2)
        arrow_head_right_y = arrow_head_y + arrow_hear_width * math.sin(angle - math.pi / 2)

        # draw the arrow head
        pygame.draw.polygon(screen, color, (end_pos, (arrow_head_left_x, arrow_head_left_y), (arrow_head_right_x, arrow_head_right_y)))