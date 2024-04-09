
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# DIMPLE params
noises = [  2,  4,   8,  10]
cycles = [50, 100, 150, 200]
num_tests = 5

# Parse data from CSV
data = [[[],[]] for _ in noises]

noise_index = 0
cycle_index = 0
with open('data_noise.csv', newline = "\n") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        if row == []:
            noise_index += 1
            cycle_index = 0
        else:
            for score in row:
                if score != "":
                    data[noise_index][0].append(int(float(score)))
                    data[noise_index][1].append(cycles[cycle_index])
            cycle_index += 1

#print(data)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hlines(0, 0, 200, color = "black", linestyles="dashed")

color = iter(plt.cm.gist_rainbow(np.linspace(0, 1, len(noises))))

for i in range(len(data)):
    c = next(color)
    ax.scatter(data[i][1], data[i][0], color=c, s = 100-(30*i))

#ax.legend()

ax.set_ylabel("Accuracy")
ax.set_xlabel("Settling Time")

plt.show()
