
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# DIMPLE params
noises = [  5, 10, 15, 20 ]
cycles = [  50, 100, 150, 200, 250, 300]
num_tests = 10

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

# Get distance between images
checker_1d = checker.reshape(64)
box_1d  = box.reshape(64)
check_dist = np.sum(np.abs(checker_1d - box_1d))
check_score = 1 - (check_dist/64)

# Parse data from CSV
data = [[[],[]] for _ in noises]
avg  = [[[],[]] for _ in noises]
low  = [[] for _ in noises]
high = [[] for _ in noises]

noise_index = 0
cycle_index = 0

for i in range(len(noises)):
    data[i][0].append(1 - (float(2*noises[i])/64))
    data[i][1].append(0)
    avg[i][0].append(1 - (float(2*noises[i])/64))
    avg[i][1].append(0)

with open('data_noise.csv', newline = "\n") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        if row == []:
            noise_index += 1
            cycle_index = 0
        else:
            if cycles[cycle_index] != 0:
                total = 0
                for score in row:
                    if score != "":
                        data[noise_index][0].append(1 - (float(score)/64))
                        data[noise_index][1].append(cycles[cycle_index] * 4)
                        total += 1 - float(score)/64
                avg_val = total/num_tests
                avg[noise_index][0].append(avg_val)
                avg[noise_index][1].append(cycles[cycle_index] * 4)
                std = stat.pstdev(data[noise_index][0])
                low[noise_index].append(avg_val - std)
                high[noise_index].append(avg_val + std)
            cycle_index += 1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hlines(1, 0, 300*4, color = "black", linestyles="dashed")
ax.hlines(check_score, 0, 300*4, color = "red", linestyles="dashed")

color = iter(plt.cm.cool(np.linspace(0, 1, len(noises))))

for i in range(len(data)):
    c = next(color)
    ax.scatter(data[i][1], data[i][0], color=c, s = 10*(2**(len(data) - i)))
    ax.plot(avg[i][1], avg[i][0], color=c, label=f"Noise Level: {noises[i]}")

ax.legend()

ax.set_ylabel("Accuracy")
ax.set_xlabel("Settling Time (ns)")

plt.show()
