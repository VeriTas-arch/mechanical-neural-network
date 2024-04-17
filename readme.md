# Introduction to Mechanical Neural Network (MNN)

## Instructions

+ This is the simulation for Mechanical Neural Network (MNN) based on Pymunk.

### File Structure

#### Main Class

+ The HexaLattice.py works as the main part. In this file the main function is executed, which means the genetic algorithm is introduced.

#### Supporting Classes

+ In this section, some supporting classes are introduced.
+ beam.py is responsible for the creation of  flexible-stiffness beams, while node.py is reponsible for the creation of the static and float nodes.
+ operations.py contains all the external operations that are implemented in this MNN simulation, such as external force, and drawing arrows that represent the displacement.
+ In settings.py you can set the environment properties of the simulation, for example the screen size.

#### Genetic Algorithm

+ These files are relative to the execution of genetic algorithm.
+ EVA.py include functions that is needed in the training process. Note that the fitness evaluation function has two different versions, one is linear, and the other is gaussian.
+ Executer.py is used to check out the best individual got from the genetic algorithm. It is shown in the pygame window, while the training process is now invisble by remobing the pygame class.
+ Filter.py is used to plot the fitness curve of a given induvidual.

### Things to Take Care

+ If the number of nodes is too large that they overflow, you can try to configure the node radius and the beam length in settings.py.
+ The add_force function is not executed through a loop, thus the current edition of add_force in the main funciton is just a demo. It is to be reconfigured into a loop function in the next major edition.
+ Most parameters can be configured in settings.py, thus you may check out this file if you want to reset any parameter.
+ Currently the genetic algorithm has some problems that to large extent influence the result. It is to be discussed in the next section. However, with large population size the problem can be alleviated, which is not crresponding to the discussion below yet has successfully sovled this seemingly priciple problem. Therefrore you may ingnore this section for now while keeping it in mind as a deep dive into the algorithm.

## Genetic Algorithm and Related

### Problems and Discussions

+ The fitness function is not well defined, thus it is not clear how to evaluate the fitness of the individual. To be more specific, the definition of fitness fuction is an inverse process (define the target function through the trainning result, and thus define the fitness function with our target function and the trainning result) thus makes the meaning of evolution process complicated.
+ The crossover function is not well defined, in other words, not corresponding to the expected behavior of the crossover process. In a natural crossover process, the offspring will inherit the characteristics of both parents, some genes are the same as its parents while some have some changes. However, even the basis of our crossover function is troubled —— what are genes in this algorithm? A gene is an executable part of the individual, thus controlling one part of the total behavior. Nevertheless, the current crossover function is not at all in line with that. Under such circumstances, the training process should be divided into several stages, each acting as a gene.
+ The mutation function is also not well defined. Similar to the discussion above, a mutation process should at least keep some behavior of the individual unchanged. The current fuction is as well not corresponding to this principle at all.

### Possible Solutions

+ The key is to divide the training process into several stages, or genes for each individual. Each gene should be responsible for a specific behavior, and the fitness function should be defined based on the behavior of the individual.
