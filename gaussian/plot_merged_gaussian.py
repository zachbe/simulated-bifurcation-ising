
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import math

samples = np.load("gaussian_samples_rand.npy")
J = np.load("gaussian_J_rand.npy")
print(-J)
print(np.linalg.inv(-J))

num_units = 2
total_spins = 64
per = int(total_spins / num_units)

gaussian = [[] for _ in range(num_units)]

for trial in range(len(samples)):
    for unit in range(num_units):
        sam = int(np.sum(samples[trial][per * unit : per * (unit+1)]))
        gaussian[unit].append(sam)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for unit in gaussian:
    ax.hist(unit, bins=100)

cov = [[0 for c in range(num_units)] for r in range(num_units)]

for a_i in range(num_units):
    for b_i in range(num_units):
        cov[a_i][b_i] = float(np.cov(gaussian[a_i], gaussian[b_i])[0][1])

print(cov)

plt.show()
