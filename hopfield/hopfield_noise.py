import torch
import numpy as np
import matplotlib.pyplot as plt

from simulated_bifurcation import ConvergenceWarning, reset_env, set_env
from simulated_bifurcation.core import Ising
from simulated_bifurcation.optimizer import (
    SimulatedBifurcationEngine,
    SimulatedBifurcationOptimizer,
)

# Image 1: Smiley
smiley = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                   [-1, -1,  1, -1, -1,  1, -1, -1],
                   [-1, -1,  1, -1, -1,  1, -1, -1],
                   [-1, -1, -1, -1, -1, -1, -1, -1],
                   [-1,  1, -1, -1, -1, -1,  1, -1],
                   [-1,  1, -1, -1, -1, -1,  1, -1],
                   [-1, -1,  1,  1,  1,  1, -1, -1],
                   [-1, -1, -1, -1, -1, -1, -1, -1]])

# Image 2: Check
check  = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1, -1,  1, -1],
                   [-1, -1, -1, -1, -1,  1,  1, -1],
                   [-1,  1, -1, -1, -1,  1, -1, -1],
                   [-1,  1,  1, -1,  1,  1, -1, -1],
                   [-1, -1,  1,  1,  1, -1, -1, -1],
                   [-1, -1, -1,  1, -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1, -1, -1, -1]])

# Compute the weights for the hopfield network
smiley_1d = smiley.reshape(64)
check_1d  = check.reshape(64)
weights = np.zeros((64,64))

# Set weights via Hebbian learning
for i in range(64):
    for j in range(64):
        weights[i][j] = ((smiley_1d[i] * smiley_1d[j]) + (check_1d[i] * check_1d[j]))

ising = Ising(weights, digital_ising_size = 64, use_fpga = True)

# DIMPLE params
noises = [  2,  4,  6,  8, 10,  12,  14,  16,  18, 20 ]
cycles = [  0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 ]
num_tests = 10

# Make a bunch of noisy versions of smiley
smilies = [[] for _ in noises]
for i in range(len(noises)):
    for j in range(num_tests):
        noise = np.random.randint(noises[i], size=(8,8))
        smiley_noise = [[-s if n == 0 else s for s, n in zip(sm, no)] for sm, no in zip(smiley, noise)]
        smiley_noise_1d = np.array(smiley_noise).reshape(64)
        smilies[i].append(smiley_noise_1d)

# Use DIMPLE to recall smiley
scores_out = [[[] for _ in cycles] for _ in noises]
smilies_out = [[[] for _ in cycles] for _ in noises]

for k in range(len(noises)):
    for i in range(len(cycles)):
        for j in range(num_tests):
            ising.minimize(
                1,
                100000,
                use_fpga = True,
                shuffle_spins = True,
                cycles = cycles[i],
                initial_spins = smilies[k][j],
                reprogram_J = True#((i == 0) and (j == 0) and (k == 0))
            )
            smiley_out = ising.computed_spins.numpy().reshape(64)
            smilies_out[k][i] = smiley_out.reshape((8,8))
            scores_out[k][i].append(min(np.sum(np.abs(smiley_out - smiley_1d)),
                                        np.sum(np.abs(smiley_out + smiley_1d))))
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
