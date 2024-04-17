import matplotlib.pyplot as plt
import numpy as np
from settings import Settings

from pathlib import Path

path = Path(__file__).parent /'storage'/'multiprocessing'/ "fitness0.csv"
fit_data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('EVA')

ax1.plot(fit_data[:, 0], label='max fitness', linewidth=2)
ax1.plot(fit_data[:, 1], label='mean fitness', linewidth=2)

plt.legend(loc='upper right')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('EVA')


set = Settings()
N_GENERATIONS = set.N_GENERATIONS

ax2.scatter(np.arange(N_GENERATIONS), fit_data[:, 0], c='r', alpha=1, s=15, label='max fitness')
ax2.scatter(np.arange(N_GENERATIONS), fit_data[:, 1], c='b', alpha=1, s=10, label='mean fitness')

plt.legend(loc='upper right')

plt.show()
