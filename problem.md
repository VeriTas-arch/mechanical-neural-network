# Problems of Current Algorithm

## Crossover function

+ Usually a crossover happens at a certain possibility, which is determined by the crossover rate. Under such condition only some of the better-behaved individuals instead of the entire population are selected for crossover.

+ In the crossover funciton, only part of the individual ought to be replaced by the other parent. For example, the beams that connect to a certain node i will be substituted by the beams of the other individual.

## Fitness function

+ In the fitness function, expected behavior can be added as the coefficent of the fitness expression. For example, the fitness generally measures the displacement of the output edge of the network. If we want the beams connected to the static nodes have a high stiffness, it can be added as a multiplier coefficent to the displacement.

## Computation speed

+ The computation speed of the algorithm can be improved by using parallel processing. It mainly depneds on the ability of the CPU and computational memory. After removing the pygame module, such parallel processing can be achieved using the multiprocessing module in Python.
