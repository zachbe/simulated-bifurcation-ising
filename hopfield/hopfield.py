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


# Make a noisy version of smiley
noise = np.random.randint(5, size=(8,8))
smiley_noise = [[-s if n == 0 else s for s, n in zip(sm, no)] for sm, no in zip(smiley, noise)]

# Compute the weights for the hopfield network
smiley_1d = smiley.reshape(64)
check_1d  = check.reshape(64)
weights = np.zeros((64,64))

for i in range(64):
    for j in range(64):
        weights[i][j] = ((smiley_1d[i] * smiley_1d[j]) + (check_1d[i] * check_1d[j]))/2

ising = Ising(weights, digital_ising_size = 64, use_fpga = True)

# Use the simulated bifurcation algorithm to recall
ising.minimize(
    1,
    100000,
    False,
    False,
    False,
    use_window=False,
    sampling_period=50,
    convergence_threshold=50,
    use_fpga = True,
    cycles = 100000
)

smiley_out = ising.computed_spins.numpy().reshape((8,8))

# Show start images
ax1 = plt.subplot(2, 2, 1)
ax1.imshow(smiley, interpolation='nearest')
ax2 = plt.subplot(2, 2, 2)
ax2.imshow(check , interpolation='nearest')

ax3 = plt.subplot(2, 2, 3)
ax3.imshow(smiley_noise, interpolation='nearest', cmap='plasma')
ax4 = plt.subplot(2, 2, 4)
ax4.imshow(smiley_out, interpolation='nearest', cmap='plasma')

#plt.show()
plt.savefig("hopfield.png")
