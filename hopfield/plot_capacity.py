
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# DIMPLE params
trials = 10
num_images = 10

# Parse data from CSV
data  = [[] for _ in range(num_images)]
index = [[] for _ in range(num_images)]
avg   = []
avg_i = []

row_id = 1

with open('data_capacity.csv', newline = "\n") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        total = 0
        for image_id in range(row_id):
            for trial_id in range(trials):
                val = float(row[image_id*trials + trial_id])
                total += val
                data [image_id].append(val)
                index[image_id].append(float(row_id))
        avg.append(total / (len(row)-1))
        avg_i.append(row_id)
        row_id += 1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hlines(10, 1, 10, color = "red", linestyles="dashed", label = "Initialized Noise Level")

color = iter(plt.cm.gist_rainbow(np.linspace(0, 1, len(data))))

c = next(color)
ax.scatter(index[9], data[9], color=c, s = 200, label = "Noise after Settling")

for i in reversed(range(len(data)-1)):
    c = next(color)
    ax.scatter(index[i], data[i], color=c, s = 20 + (20*i))

ax.plot(avg_i, avg, color= "black", label = "Avergage Noise after Settling")

ax.legend()

ax.set_ylabel("Nosie After 1200ns Settling")
ax.set_xlabel("Number of Images Stored")

plt.show()
