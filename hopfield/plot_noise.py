
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# DIMPLE params
noises = [  2,  4,  6,  8, 10,  12,  14,  16,  18, 20 ]
cycles = [  0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 ]
num_tests = 10

# Parse data from CSV
data = [[[],[]] for _ in noises]
avg  = [[[],[]] for _ in noises]
low  = [[] for _ in noises]
high = [[] for _ in noises]

noise_index = 0
cycle_index = 0
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
                        data[noise_index][0].append(1 - float(score)/64)
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

ax.hlines(1, 20, 800, color = "black", linestyles="dashed")

color = iter(plt.cm.gist_rainbow(np.linspace(0, 1, len(noises))))

for i in range(len(data)):
    c = next(color)
    ax.scatter(data[i][1], data[i][0], color=c, s = 20)
    ax.plot(avg[i][1], avg[i][0], color=c)
#    ax.fill_between(avg[i][1], low[i], high[i], fc = c, ec = c, lw = 1, linestyles = "dashed", alpha = 0.5)

#ax.legend()

ax.set_ylabel("Accuracy")
ax.set_xlabel("Settling Time (ns)")

plt.show()
