import math

class Node:
    """class for creating nodes in the simulation"""

    def __init__(self):
        self.float_node_color = (60, 133, 166)
        
        
    # function that adds a new float node
    def add_float_node(self, space, radius, mass, pos):
        body = pymunk.Body()
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.mass = mass
        shape.color = self.float_node_color

        space.add(body, shape)
        return body

    # function that adds a static node
    def add_static_node(self, space, radius, pos):
        body = pymunk.Body(body_type = pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Circle(body, radius)

        space.add(body, shape)
        return body

    # function that initializes the nodes
    def _create_nodes(self, space):
        # calculate the position bias of the nodes, note that the size of the screen is not called automatically
        sep_h = (720 - 100) / 2
        sep_w = (1080 - 4 * 100 * math.sin(math.pi/3)) / 2
        node_list = []
        length = 20
        for _ in range(length):    
            node_list.append(None)

        for i in range(0,11):
            if i%5 == 0 or i%5 == 1:
                pos_x = sep_w + 100 * math.sin(math.pi/3) + (i%5) * 2 * 100 * math.sin(math.pi/3)
                pos_y = sep_h - 100 * math.cos(math.pi/3) + 100 * (i - i%5) / 5
                node_list[i] = self.add_float_node(space, 20, 10,(pos_x, pos_y))  

            elif i%5 == 3:
                pos_x = sep_w + 2 * 100 * math.sin(math.pi/3)
                pos_y = sep_h + 100 * (i - 3) / 5
                node_list[i] = self.add_float_node(space, 20, 10, (pos_x, pos_y))

            elif i%5 == 2 or i%5 == 4:
                pos_x = sep_w + (i%5 - 2) * 2 * 100 * math.sin(math.pi/3)
                pos_y = sep_h + 100 * (i - i%5) / 5
                node_list[i] = self.add_static_node(space, 20, 10,(pos_x, pos_y))

            


         