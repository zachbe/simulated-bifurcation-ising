
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import math


# Read in data and print expected values
samples = np.load("gaussian_samples_rand.npy")
J = np.load("gaussian_J_rand.npy")
print("Input matrix")
print(-J)
inv = np.linalg.inv(-J)
print("Target inverse")
print(inv/inv[0][0])

h = np.load("gaussian_h_rand.npy")
print("Input vector")
print(-h)
sol = inv @ (-h)
print("Linear system solution")
print(sol/sol[0])

print("--------------------------")

# Paramters for data collation
num_bits = 2
num_units = 3
total_spins = 64
per = int(total_spins / (num_units * num_bits))

# Collate data into buckets
gaussian = [[0 for s in range(len(samples))] for _ in range(num_units)]
for trial in range(len(samples)):
    for unit in range(num_units):
        for bit in range(num_bits):
            sam = (2**bit) * int(np.sum(samples[trial][unit*per*num_bits + per*bit : unit*per*num_bits + per*(bit+1)]))
            gaussian[unit][trial] += sam

# Calculate mean and covariance
cov = [[0 for c in range(num_units)] for r in range(num_units)]
for a_i in range(num_units):
    for b_i in range(num_units):
        cov[a_i][b_i] = float(np.cov(gaussian[a_i], gaussian[b_i])[0][1])
print("Device inverse")
print(np.array(cov)/cov[0][0])

mean = [0 for _ in range(num_units)]
for a in range(num_units):
    mean[a] = np.sum(gaussian[a])/per
print("Device linear system solution")
print(mean/mean[0])

# Plot hisograms
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for unit in gaussian:
    ax.hist(unit, bins=100)

plt.show()
