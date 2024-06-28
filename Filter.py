import math
from pathlib import Path

import EVA
import matplotlib.pyplot as plt
import numpy as np
import pymunk
from beam import Beam
from node import Node
from operations import Operations
from settings import Settings


class HexaLattice:
    """Main class for HexaLattice simulation"""

    def __init__(self, stiffness_mat):
        """program initialization"""
        self.settings = Settings()

        # initialize pymunk
        self.space = pymunk.Space()
        self.space.gravity = self.settings.gravity

        """initialize external classes"""
        self.beam = Beam(self)
        self.node = Node(self)
        self.operations = Operations(self)

        """define the constants"""
        # simulation parameters
        self.length = self.settings.length
        self.step = self.settings.step
        self.row_lenh = self.settings.row_lenh
        self.row_num = self.settings.row_num

        # node and beam parameters
        self.blen = self.settings.beam_length
        self.T = 2 * self.row_lenh + 1
        self.radius = self.settings.node_radius
        self.mass = self.settings.float_node_mass

        # execution parameters
        self.stiffness_mat = stiffness_mat
        self.step_counter = 0
        self.step_interval = 500

        """initialize the lists"""
        self.node_list = [None for i in range(self.length)]
        self.beam_list = []
        self.init_pos = [None for i in range(self.length)]
        self.dynamic_pos = [None for i in range(self.length)]
        self.node_record = [None for i in range(self.length)]

        for i in range(self.length):
            self.beam_list.append([None for j in range(self.length)])

        # create the nodes and beams
        self._init_pos()
        self._create_nodes(self.space)
        self._create_beams(self.stiffness_mat)

        # create the fitness plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness Curve")

        self.running = True

    def run_game(self):
        """Main game loop"""
        # while self.running:
        for _ in range(self.step_interval):
            self._update_screen()
            self.step_counter += 1
            self.space.step(self.step)

            # record the dynamic position of the nodes and draw the fitness curve
            self._update_pos()
            self._plot_fitness()

    def _update_screen(self):
        """Update the screen"""

        # apply force to the nodes
        body_1, body_2 = self.node_list[0], self.node_list[1]
        (force_1_x, force_1_y) = self.settings.force_1
        (force_2_x, force_2_y) = self.settings.force_2
        self.operations.add_force(body_1, force_1_x, force_1_y)
        self.operations.add_force(body_2, force_2_x, force_2_y)

        # apply frictions
        for i in range(0, self.settings.length):
            v = math.sqrt(
                self.node_list[i].velocity[0] ** 2 + self.node_list[i].velocity[1] ** 2
            )
            f = self.settings.friction
            F = math.sqrt(
                self.node_list[i].force[0] ** 2 + self.node_list[i].force[1] ** 2
            )

            if v >= 1e-2:
                e = self.node_list[i].velocity / v
                friction_x, friction_y = -f * e[0], -f * e[1]
            elif v < 1e-2 and F < f:
                friction_x, friction_y = (
                    -self.node_list[i].force[0],
                    -self.node_list[i].force[1],
                )
            else:
                e = self.node_list[i].force / F
                friction_x, friction_y = -f * e[0], -f * e[1]

            self.operations.add_force(self.node_list[i], friction_x, friction_y)

    def _init_pos(self):
        """calculate the initial position of the nodes"""
        sep_x = (
            self.settings.screen_width - (self.row_lenh - 1) * self.blen * math.sqrt(3)
        ) / 2
        sep_y = (self.settings.screen_height - ((self.row_num - 1) / 2) * self.blen) / 2

        n = self.row_lenh
        column_counter = 0
        row_counter = 0

        for i in range(self.length):
            if column_counter == n or column_counter == self.T:
                column_counter = column_counter % self.T
                row_counter += 1

            if column_counter < n:
                pos_x = sep_x + column_counter * self.blen * math.sqrt(3)
                pos_y = sep_y + row_counter * self.blen / 2
            else:
                pos_x = (
                    sep_x
                    + (column_counter - n) * self.blen * math.sqrt(3)
                    - self.blen * math.sqrt(3) / 2
                )
                pos_y = sep_y + row_counter * self.blen / 2

            column_counter += 1
            self.init_pos[i] = (pos_x, pos_y)

    def _create_nodes(self, space):
        """function that initializes the nodes"""
        for i in range(self.length):
            if (i + 1) % self.T == 0 or (i + 1) % self.T == self.row_lenh + 1:
                self.node_record[i] = self.node.add_static_node(
                    space, self.radius, self.init_pos[i]
                )
            else:
                self.node_record[i] = self.node.add_float_node(
                    space, self.radius, self.mass, self.init_pos[i]
                )

            self.node_list[i] = self.node_record[i][0]

    def _create_beams(self, stiffness_mat):
        """function that initializes the beams"""
        n = self.row_lenh
        # notion_mat = np.zeros((length, length))

        for i in range(self.length):
            for j in range(i + 1, self.length):
                if i % (2 * n + 1) != n and i % (2 * n + 1) != 2 * n:
                    if j == i + n or j == i + n + 1 or j == i + 2 * n + 1:
                        # notion_mat[i][j] = 1
                        self.beam_list[i][j] = self.beam.add_beam(
                            self.node_list[i], self.node_list[j], stiffness_mat[i][j]
                        )

                elif i % (2 * n + 1) == n and j == i + n + 1:
                    # notion_mat[i][j] = 1
                    self.beam_list[i][j] = self.beam.add_beam(
                        self.node_list[i], self.node_list[j], stiffness_mat[i][j]
                    )

                elif i % (2 * n + 1) == 2 * n and j == i + n:
                    # notion_mat[i][j] = 1
                    self.beam_list[i][j] = self.beam.add_beam(
                        self.node_list[i], self.node_list[j], stiffness_mat[i][j]
                    )

        # print(notion_mat)
        # return notion_mat

    def _update_pos(self):
        """record the dynamic position of the nodes"""
        for i in range(self.length):
            self.dynamic_pos[i] = self.node_list[i].position

    def _plot_fitness(self):
        """calculate the fitness"""
        fitness = EVA.get_fitness(self.dynamic_pos)
        # print("the fitness is:", EVA.get_fitness(pop_pos))
        self.ax.scatter(self.step_counter, fitness, c="r", s=10)
        plt.draw()
        plt.pause(0.01)


if __name__ == "__main__":
    # read the stiffness matrix from the csv file
    path = Path(__file__).parent / "individual.csv"
    stiffness_mat = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)

    # create the HexaLattice object
    hexalattice = HexaLattice(stiffness_mat)

    """run the simulation"""
    hexalattice.run_game()
    print(EVA.get_fitness(hexalattice.dynamic_pos))
    plt.show()
