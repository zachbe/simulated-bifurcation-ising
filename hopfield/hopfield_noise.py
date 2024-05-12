import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from simulated_bifurcation import ConvergenceWarning, reset_env, set_env
from simulated_bifurcation.core import Ising
from simulated_bifurcation.optimizer import (
    SimulatedBifurcationEngine,
    SimulatedBifurcationOptimizer,
)

# Image 1: Checker
checker   = np.array([[ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1]])

# Image 2: Box
box       = np.array([[ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1]])

# Compute the weights for the hopfield network
checker_1d = checker.reshape(64)
box_1d  = box.reshape(64)
weights = np.zeros((64,64))

# Set weights via Hebbian learning
for i in range(64):
    for j in range(64):
        weights[i][j] = ((checker_1d[i] * checker_1d[j]) + (box_1d[i] * box_1d[j]))

ising = Ising(weights, digital_ising_size = 64, use_fpga = True)

# DIMPLE params
noises = [  5, 10, 15, 20 ]
cycles = [  50, 100, 150, 200, 250, 300]
num_tests = 10

# Make a bunch of noisy versions of checker
checkers = [[] for _ in noises]
for i in range(len(noises)):
    for j in range(num_tests):
        noise = np.zeros(64)
        while np.sum(noise) < noises[i]:
            noise[random.randint(0,63)] = 1
        checker_noise_1d = [-s if n == 0 else s for s, n in zip(checker_1d, noise)]
        checkers[i].append(checker_noise_1d)

# Use DIMPLE to recall checker
scores_out = [[[] for _ in cycles] for _ in noises]
checkers_out = [[[] for _ in cycles] for _ in noises]

for k in range(len(noises)):
    for i in range(len(cycles)):
        for j in range(num_tests):
            ising.minimize(
                1,
                100000,
                use_fpga = True,
                shuffle_spins = True,
                cycles = cycles[i],
                initial_spins = checkers[k][j],
                reprogram_J = True
            )
            checker_out = ising.computed_spins.numpy().reshape(64)
            checkers_out[k][i] = checker_out.reshape((8,8))
            scores_out[k][i].append(min(np.sum(np.abs(checker_out - checker_1d)),
                                        np.sum(np.abs(checker_out + checker_1d))))
            print(j)
        print("Cycles: " + str(cycles[i]))
    print("Noises: " + str(noises[k]))


f = open("data_noise.csv", "w+")
for n in scores_out:
    for c in n:
        for t in c:
            f.write(str(t)+",")
        f.write("\n")
    f.write("\n")
