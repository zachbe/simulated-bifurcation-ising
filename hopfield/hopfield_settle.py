import torch
import numpy as np
import matplotlib.pyplot as plt

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

# Make a noisy version of checker
noise = np.random.randint(4, size=(8,8))
checker_noise = [[-s if n == 0 else s for s, n in zip(sm, no)] for sm, no in zip(checker, noise)]
checker_noise = np.array(checker_noise)
checker_noise_1d = checker_noise.reshape(64)

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
cycles = [50, 100, 150, 200]

# Use DIMPLE to recall checker in steps
checkers_out = [checker_noise]

for i in range(len(cycles)):
    ising.minimize(
        1,
        100000,
        use_fpga = True,
        shuffle_spins = False,
        cycles = cycles[i],
        initial_spins = checker_noise_1d,
        reprogram_J = (i == 0)
    )
    checkers_out.append(ising.computed_spins.numpy().reshape((8,8)))

cycles.insert(0,0)

# Show images
for i in range(len(cycles)):
    ax = plt.subplot(1, len(cycles), i+1)
    ax.imshow(checkers_out[i], interpolation='nearest')
    sec = cycles[i]*4
    ax.title.set_text(f"{sec} ns")
    ax.axis("off")

#plt.show()
plt.tight_layout()
plt.savefig("hopfield_settle.png", dpi=200)
