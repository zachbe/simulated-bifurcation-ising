
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# Data should contain num_tests tests at each different coupling density.
# For each test, there will have been num_trials trials.

if (len(sys.argv) != 4):
    print("should be run as:") 
    print("sudo python3 plot_data_time.py [num_tests] [num_trials] [num_time_sweep_points]")
    sys.exit() 

num_tests = int(sys.argv[1])
num_trials = int(sys.argv[2])
num_den_sweep = int(sys.argv[3])

energy   = [[] for _ in range(50)]
sparsity = [[] for _ in range(50)]

all_spar = {}

row_id = 0

# Parse data from CSV
with open('data_time.csv', newline = "\n") as data:
    reader = csv.reader(data, delimiter=",")
    for row in reader:
        sim_energy = float(row[-2])
        for i in range(0, num_trials, 2):
            fpga_energy = float(row[i])/sim_energy #normalize
            time = float(row[i+1])
            index = row_id
            energy[index].append(fpga_energy)
            sparsity[index].append(time)

            #if sparsity_row in all_spar:
            #    all_spar[sparsity_row].append(fpga_energy)
            #else:
            #    all_spar[sparsity_row] = [fpga_energy]
        row_id += 1

# Calculate average solution quality + error bars
means = []
spars = []
lows  = []
highs = []
for spar_val in all_spar.keys():
    spar_en = all_spar[spar_val]
    spars.append(spar_val)
    mean = sum(spar_en) / len(spar_en)
    std = stat.pstdev(spar_en)
    means.append(mean)
    lows.append(mean - std)
    highs.append(mean + std)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

color = iter(plt.cm.Set1(np.linspace(0, 1, len(energy))))

c = next(color)
ax.scatter(sparsity[0], energy[0], color=c, s = 20, label = "Solution Attempt")

for (en, sp) in zip(energy[1:], sparsity[1:]):
    c = next(color)
    ax.scatter(sp, en, color=c, s = 20)

#ax.fill_between(spars, lows, highs, fc = (1, 0, 0, 0.5), ec = (0.5,0,0,1), lw=1, linestyles = "dashed", label = "Std Dev, Solution Quality")
#ax.plot(spars, means, color = "black", label = "Average Normalized Solution Quality")
ax.hlines(1, 0.002, 0.003, color = "black", linestyles="dashed")
ax.legend()

ax.set_ylabel("Normalized Solution Quality")
ax.set_xlabel("Time-to-Solution (sec)")
ax.set_xlim([0.002, 0.003])
#ax.set_yscale('log')

plt.show()
