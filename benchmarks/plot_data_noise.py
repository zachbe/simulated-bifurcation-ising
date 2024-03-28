
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# For each test, there will have been num_trials trials.

if (len(sys.argv) != 3):
    print("should be run as:") 
    print("sudo python3 plot_data.py [num_tests] [num_trials]")
    sys.exit() 

num_tests = int(sys.argv[1])
num_trials = int(sys.argv[2])

energy    = [[] for _ in range(num_tests)]
row_index = [[] for _ in range(num_tests)]

all_spar = {}

row_id = 0

# Parse data from CSV
with open('data_noise.csv', newline = "\n") as data:
    reader = csv.reader(data, delimiter=",")
    for row in reader:
        sim_energy = float(row[0])
        for i in range(2, 2 + num_trials, 2):
            fpga_energy = float(row[i])/sim_energy #normalize
            index = row_id % num_tests
            energy[index].append(fpga_energy)
            row_index[index].append(row_id)

        row_id += 1


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

color = iter(plt.cm.tab10(np.linspace(0, 1, num_tests)))

c = next(color)
ax.scatter(row_index[0], energy[0], color=c, s = 20, label = "Solution Attempt")

for (en, sp) in zip(energy[1:], row_index[1:]):
    c = next(color)
    ax.scatter(sp, en, color=c, s = 20)

ax.hlines(1, 0, 10, color = "black", linestyles="dashed")
#ax.set_ylim([0.5, 1.2])
ax.legend()

ax.set_ylabel("Normalized Solution Quality")
ax.set_xlabel("J Matrix")
#ax.set_yscale('log')

plt.show()
