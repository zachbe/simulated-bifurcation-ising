
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import math

samples = np.load("gaussian_samples_rand.npy")

a = []
b = []

for trial in range(len(samples)):
    a_sam = int(np.sum(samples[trial][0:32]))
    b_sam = int(np.sum(samples[trial][32:63]))
    a.append(a_sam)
    b.append(b_sam)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(a, bins=100)
ax.hist(b, bins=100)

print(np.cov(a,b))

plt.show()
