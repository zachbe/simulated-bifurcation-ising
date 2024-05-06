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

images = [[] for _ in range(10)]

# Image 1: Checker
images[0] = np.array([[ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1]])

# Image 2: Box
images[1] = np.array([[ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1]])

# Image 3: All 1
images[2] = np.array([[ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1]])

# Image 4: Horizontal Box
images[3] = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1]])

# Image 5: Small Checker
images[4] = np.array([[-1, -1,  1,  1, -1, -1,  1,  1],
                      [-1, -1,  1,  1, -1, -1,  1,  1],
                      [ 1,  1, -1, -1,  1,  1, -1, -1],
                      [ 1,  1, -1, -1,  1,  1, -1, -1],
                      [-1, -1,  1,  1, -1, -1,  1,  1],
                      [-1, -1,  1,  1, -1, -1,  1,  1],
                      [ 1,  1, -1, -1,  1,  1, -1, -1],
                      [ 1,  1, -1, -1,  1,  1, -1, -1]])

# Image 6: Verical Blinds
images[5] = np.array([[-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1]])

# Image 7: Horizontal Blinds
images[6] = np.array([[-1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [-1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1]])

# Image 8: Total Checker
images[7] = np.array([[ 1, -1,  1, -1,  1, -1,  1, -1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1],
                      [-1,  1, -1,  1, -1,  1, -1,  1]])

# Image 9: Split Horizontal Blinds
images[8] = np.array([[-1, -1, -1, -1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1],
                      [-1, -1, -1, -1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1, -1, -1, -1, -1]])

# Image 10: Split Vertical Blinds
images[9] = np.array([[-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [-1,  1, -1,  1, -1,  1, -1,  1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1],
                      [ 1, -1,  1, -1,  1, -1,  1, -1]])

# Show images
for i in range(10):
    ax = plt.subplot(1, 10, i+1)
    ax.imshow(images[i], interpolation='nearest')
    ax.axis("off")

plt.tight_layout()
plt.savefig("hopfield_10.png", dpi=200)

scores = [[] for _ in range(10)]

for num_images in range(10):
    # Compute the weights for the hopfield network
    # Set weights via Hebbian learning
    images_1d = []
    for i in range(num_images+1):
        image = images[i]
        images_1d.append(image.reshape(64))
    weights = np.zeros((64,64))
    
    for i in range(64):
        for j in range(64):
            for image in images_1d:
                weights[i][j] += (image[i] * image[j])
    
    ising = Ising(weights, digital_ising_size = 64, use_fpga = True)
    
    # DIMPLE params
    cycles = 300
    trials = 5
    
    for image in images_1d: 
        for trial in range(trials):
            noise = np.zeros(64)
            while np.sum(noise) < 5:
                noise[random.randint(0,63)] = 1
            image_noise = [-s if n == 0 else s for s, n in zip(image, noise)]
            ising.minimize(
                1,
                100000,
                use_fpga = True,
                shuffle_spins = True,
                cycles = cycles,
                initial_spins = image_noise,
                reprogram_J = True
            )
            image_out = ising.computed_spins.numpy().reshape(64)
            score = min(np.sum(np.abs(image_out - image)), np.sum(np.abs(image_out + image)))
            scores[num_images].append(score)
            print(f"{num_images} {trial}")

f = open("data_capacity.csv", "w+")
for n in scores:
    for t in n:
        f.write(str(t)+",")
    f.write("\n")
