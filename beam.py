class Beam:
    """class for creating beams in the simulation"""
    
    def __init__(self):
        self.damping = 10

    # fuction that adds a spring (i.e. beam)
    def add_beam(self, body_1, body_2, stiffness):
        damping = self.damping
        spring = pymunk.DampedSpring(body_1, body_2, (0, 0), (0, 0), 0, stiffness, damping)
        
        space.add(spring)