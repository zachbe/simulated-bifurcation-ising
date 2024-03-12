
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

# Data should contain num_tests tests at each different coupling density.
# For each test, there will have been num_trials trials.

if (len(sys.argv) != 4):
    print("should be run as:") 
    print("sudo python3 plot_data.py [num_tests] [num_trials] [num_density_sweep_points]")
    sys.exit() 

num_tests = int(sys.argv[1])
num_trials = int(sys.argv[2])
num_den_sweep = int(sys.argv[3])

energy   = [[] for _ in range(num_tests)]
sparsity = [[] for _ in range(num_tests)]

row_id = 0

with open('data.csv', newline = "\n") as data:
    reader = csv.reader(data, delimiter=",")
    for row in reader:
        sparsity_row = float(row[0])
        sim_energy = float(row[1])
        for i in range(2, 2 + num_trials):
            fpga_energy = float(row[i])/sim_energy #normalize
            index = row_id % num_tests
            energy[index].append(fpga_energy)
            sparsity[index].append(1-sparsity_row)
        row_id += 1

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

color = iter(plt.cm.gist_rainbow(np.linspace(0, 1, num_tests)))

for (en, sp) in zip(energy, sparsity):
    c = next(color)
    ax.scatter(sp, en, color=c, s = 10)

ax.hlines(1, 0, 1, color = "black")

ax.set_ylabel("Normalized Solution Energy")
ax.set_xlabel("Graph Density")
#ax.set_yscale('log')

plt.show()
