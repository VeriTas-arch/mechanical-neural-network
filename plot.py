from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from settings import Settings

set = Settings()
N_CORES = set.N_CORES
N_GENERATIONS = set.N_GENERATIONS
POP_SIZE = set.POP_SIZE

crossover_default = np.ones(N_CORES) * 0.6
mutation_default = np.ones(N_CORES) * 0.1
rate_default = np.array([crossover_default, mutation_default])


def plot_fitness(rate_default):
    for i in range(N_CORES):
        crossover_rate = rate_default[i][0]
        mutation_rate = rate_default[i][1]

        path = (
            Path(__file__).parent
            / "storage"
            / "multiprocessing"
            / "fitness_data"
            / f"fitness{i}.csv"
        )
        fit_data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=0)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        plt.xlabel(
            f"Generation (pop_size: {POP_SIZE}, net_size: ({set.col_lenh},{set.row_lenh}))"
        )
        plt.ylabel("Fitness")
        plt.title(f"crossover: {crossover_rate:.2f}, mutation: {mutation_rate:.2f}")

        ax1.plot(fit_data[:, 0], label="max fitness", linewidth=2)
        ax1.plot(fit_data[:, 1], label="mean fitness", linewidth=2)

        plt.legend(loc="lower right")

        path1 = (
            Path(__file__).parent
            / "storage"
            / "multiprocessing"
            / "figures"
            / f"fitness{i}_1.png"
        )
        fig1.savefig(path1)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        plt.xlabel(
            f"Generation (pop_size: {POP_SIZE}, net_size: ({set.col_lenh},{set.row_lenh}))"
        )
        plt.ylabel("Fitness")
        plt.title(f"crossover: {crossover_rate:.2f}, mutation: {mutation_rate:.2f}")

        len_fit_data = len(fit_data)

        ax2.scatter(
            np.arange(len_fit_data),
            fit_data[:, 0],
            c="r",
            alpha=1,
            s=15,
            label="max fitness",
        )
        ax2.scatter(
            np.arange(len_fit_data),
            fit_data[:, 1],
            c="b",
            alpha=1,
            s=10,
            label="mean fitness",
        )

        plt.legend(loc="lower right")

        path2 = (
            Path(__file__).parent
            / "storage"
            / "multiprocessing"
            / "figures"
            / f"fitness{i}_2.png"
        )
        fig2.savefig(path2)
