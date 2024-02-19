#### Instructions

+ This is the simulation for Mechanical Neural Network (MNN) based on Pymunk
+ For the five python files, the HexaLattice.py works as the main part. In this file the main function is executed.
+ beam.py is responsible for the creation of  flexible-stiffness beams, while node.py is reponsible for the creation of the static and float nodes.
+ operations.py contains all the external operations that are implemented in this MNN simulation, such as external force, and drawing arrows that represent the displacement.
+ In settings.py you can set the environment properties of the simulation, for example the screen size.

#### Things to Take Care

+ (At present, there is nothing major that needs extra attention)
+ If the number of nodes is too large that they overflow, you can try to configure the node radius and the beam length in settings.py.
+ The add_force function is not executed through a loop, thus the current edition of add_force in the main funciton is just a demo. It is to be reconfigured into a loop function in the next major edition.
+ Most parameters can be configured in settings.py, thus you may check out this file if you want to reset any parameter.
