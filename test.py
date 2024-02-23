from settings import *
import numpy as np

parent = np.random.randn(11,11)
pop = np.random.randn(11,11)

crossover_result = []

for i in range(11):
    crossover_result.append([])

for i in range(11):
    for j in range(11):
        if j <3:
            crossover_result[i].append(parent[i][j])
        else:
            crossover_result[i].append(pop[2][j])

print(crossover_result[0])