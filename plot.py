import matplotlib.pyplot as plt
import numpy as np
from settings import Settings

from pathlib import Path

set = Settings()
N_CORES = set.N_CORES
N_GENERATIONS = set.N_GENERATIONS
POP_SIZE = set.POP_SIZE


def plot_fitness(crossover_rate=0.6, mutation_rate=0.1):
    for i in range(N_CORES):
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

        ax2.scatter(
            np.arange(N_GENERATIONS),
            fit_data[:, 0],
            c="r",
            alpha=1,
            s=15,
            label="max fitness",
        )
        ax2.scatter(
            np.arange(N_GENERATIONS),
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
