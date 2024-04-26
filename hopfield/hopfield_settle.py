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
noise = np.random.randint(10, size=(8,8))
smiley_noise = smiley #[[-s if n == 0 else s for s, n in zip(sm, no)] for sm, no in zip(smiley, noise)]
smiley_noise = np.array(smiley_noise)
smiley_noise_1d = smiley_noise.reshape(64)

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
cycles = [20, 40, 60, 80, 100, 200, 400, 800, 1000]

# Use DIMPLE to recall smiley in steps
smilies_out = [smiley_noise]

for i in range(len(cycles)):
    ising.minimize(
        1,
        100000,
        use_fpga = True,
        shuffle_spins = False,
        cycles = cycles[i],
        initial_spins = smiley_noise_1d,
        reprogram_J = (i == 0)
    )
    smilies_out.append(ising.computed_spins.numpy().reshape((8,8)))

cycles.insert(0,0)

# Show images
for i in range(len(cycles)):
    ax = plt.subplot(1, len(cycles), i+1)
    ax.imshow(smilies_out[i], interpolation='nearest')
    sec = cycles[i]*4
    ax.title.set_text(f"{sec} ns")
    ax.axis("off")

#plt.show()
plt.tight_layout()
plt.savefig("hopfield_settle.png", dpi=200)
