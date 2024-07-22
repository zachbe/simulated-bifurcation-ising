
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import math

samples = np.load("gaussian_samples_rand.npy")

a = []
b = []
c = []

print(len(samples))

for trial in range(len(samples)):
    a_sam = int(np.sum(samples[trial][ 0:21]))
    b_sam = int(np.sum(samples[trial][21:42]))
    c_sam = int(np.sum(samples[trial][42:63]))
    a.append(a_sam)
    b.append(b_sam)
    c.append(c_sam)
    if (a_sam != -21 or b_sam != -21 or c_sam != -21):
        print(a_sam)
        print(b_sam)
        print(c_sam)
        print("-------")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(a, bins=100)
ax.hist(b, bins=100)

print(np.cov(a,b))
print(np.cov(b,c))
print(np.cov(a,c))

plt.show()
