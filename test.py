import numpy as np

parent = np.random.randn(11, 11)
pop = np.random.randn(11, 11)

crossover_result = []

for i in range(11):
    crossover_result.append([])

for i in range(11):
            crossover_result[i].extend(parent[i][:3])
            crossover_result[i].extend(pop[2][3:])

print(crossover_result[0])
