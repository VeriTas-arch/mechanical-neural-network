# Introduction to Mechanical Neural Network (MNN)

## Instructions

+ This is the simulation for Mechanical Neural Network (MNN) based on Pymunk.
+ In our simulation, a spring-mass system is applied to simplify the representation of Mechanical Neural Network, yet some properties may get lost in this procedure compared to the original version.

### File Structure

```shell
.
├── src
│   ├── HexaLattice.py
│   ├── EVA.py, plot.py
│   ├── node.py, beam.py, operations.py 
│   └── settings.py
├── assets
│   ├── figures
│   └── fitness_data
├── .gitignore
└── README.md
```

#### Main Class

+ The `HexaLattice.py` works as the main part. In this file the simulation function `pymunk_run()` is executed, where the genetic algorithm is introduced.

#### Genetic Algorithm

+ These files are relative to the execution of genetic algorithm.
+ `EVA.py` includes functions that are needed in the training process. Note that the crossover function now includes two types, but the second one is nearly abandoned (which was inherited from older versions, yet proved inefficient) .
+ `plot.py` includes a function that uses the training results to draw the fitness-generation pictures.
+ `Executer.py` is used to check out the best individual got from the genetic algorithm. It is shown in the pygame window, while the training process is now invisble by removing the pygame class.
+ `Filter.py` is used to plot the fitness curve of a given induvidual. Thus the convergence simulation step can be estimated.

#### Supporting Classes

+ In this section, some supporting classes are introduced.
+ `beam.py` is responsible for the creation of flexible-stiffness beams, while `node.py` is reponsible for the creation of the static and float nodes.
+ `operations.py` contains all the external operations that are implemented in this MNN simulation, such as external force, and drawing arrows that represent the displacement.
+ In `settings.py` you can set the environment properties of the simulation, for example the screen size.

### Things to Take Care

+ If the number of nodes is too large that they overflow, you can try to configure the node radius and the beam length in `settings.py`.
+ Most parameters can be configured in `settings.py`, thus you may check out this file if you want to reset any parameter.
+ Currently the genetic algorithm has some problems that to large extent influence the result. It is yet unknown to us whether such problem results from the algorithm or the network structure itself. However, with large population size the problem can be alleviated, which is not crresponding to the discussion below yet has successfully sovled this seemingly principle problem under certain conditions. Therefrore you may ingnore this section for now while keeping it in mind as a deep dive into the algorithm.

## Genetic Algorithm and Related

### Problems and Discussions

+ The fitness function is not well defined, thus it is not clear how to evaluate the fitness of the individual. To be more specific, the definition of fitness fuction is an inverse process (define the target function through the trainning result, and thus define the fitness function with our target function and the trainning result) thus makes the meaning of evolution process complicated.
+ The crossover function is not well defined, in other words, not corresponding to the expected behavior of the crossover process. In a natural crossover process, the offspring will inherit the characteristics of both parents, some genes are the same as its parents while some have some changes. However, even the basis of our crossover function is troubled —— what are genes in this algorithm? A gene is an executable part of the individual, thus controlling one part of the total behavior. Nevertheless, the current crossover function is not at all in line with that. Under such circumstances, the training process should be divided into several stages, each acting as a gene.
+ The mutation function is also not well defined. Similar to the discussion above, a mutation process should at least keep some behavior of the individual unchanged. The current fuction is as well not corresponding to this principle at all.

### Possible Solutions

+ The key is to divide the training process into several stages, or genes for each individual. Each gene should be responsible for a specific behavior, and the fitness function should be defined based on the behavior of the individual.
+ We have noticed that the genetic algorithmperforms a pretty slow training, while other algorithms such as PPS have behaved better. After the most parts of genetic algorithm are clear, other algorithms will be put into schedule.
