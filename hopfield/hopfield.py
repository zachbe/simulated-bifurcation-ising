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
noise = np.random.randint(20, size=(8,8))
smiley_noise = [[s if n == 0 else -s for s, n in zip(sm, no)] for sm, no in zip(smiley, noise)]
smiley_noise = np.array(smiley_noise)
smiley_noise_1d = smiley_noise.reshape(64)

# Make a noisy version of check
noise = np.random.randint(20, size=(8,8))
check_noise = [[c if n == 0 else -c for c, n in zip(ch, no)] for ch, no in zip(check, noise)]
check_noise = np.array(check_noise)
check_noise_1d = check_noise.reshape(64)

# Compute the weights for the hopfield network
smiley_1d = smiley.reshape(64)
check_1d  = check.reshape(64)
weights = np.zeros((64,64))

# Set weights via Hebbian learning
for i in range(64):
    for j in range(64):
        weights[i][j] = ((smiley_1d[i] * smiley_1d[j]) + (check_1d[i] * check_1d[j]))

ising = Ising(weights, digital_ising_size = 64, use_fpga = True)

# Use DIMPLE to recall smiley
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
    shuffle_spins = True,
    cycles = 1000000,
    initial_spins = smiley_noise_1d
)
smiley_out = ising.computed_spins.numpy().reshape((8,8))

# Use DIMPLE to recall check
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
    shuffle_spins = True,
    cycles = 1000000,
    initial_spins = check_noise_1d
)
check_out = ising.computed_spins.numpy().reshape((8,8))

# Show start images
ax1 = plt.subplot(3, 2, 1)
ax1.imshow(smiley, interpolation='nearest')
ax2 = plt.subplot(3, 2, 2)
ax2.imshow(check , interpolation='nearest')

ax3 = plt.subplot(3, 2, 3)
ax3.imshow(smiley_noise, interpolation='nearest', cmap='plasma')
ax4 = plt.subplot(3, 2, 4)
ax4.imshow(smiley_out, interpolation='nearest', cmap='plasma')

ax5 = plt.subplot(3, 2, 5)
ax5.imshow(check_noise, interpolation='nearest', cmap='cividis')
ax6 = plt.subplot(3, 2, 6)
ax6.imshow(check_out, interpolation='nearest', cmap='cividis')

#plt.show()
plt.savefig("hopfield.png")
