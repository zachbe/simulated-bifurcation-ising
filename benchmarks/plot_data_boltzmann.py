
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import math

# Data should contain num_tests tests at each different coupling density.
# For each test, there will have been num_trials trials.

if (len(sys.argv) != 2):
    print("should be run as:") 
    print("sudo python3 plot_data_boltzmann.py [num_samples] [num_trials]")
    sys.exit() 

expected_energy = []
expected_prob   = []
total = 0
# Generate expected distribution
for i in range(127):
    # There are 62 total couplings. Each can be +7 or -7.
    # Thus, there are a total of 62 possible energies.
    state_energy = (i * 7) + ((127-i) * -7)
    probability = math.exp(-state_energy/9)
    density = math.comb(127,i)
    expected_energy.append(state_energy)
    expected_prob.append(probability * density)
    total += probability * density

expected_prob = [_/total for _ in expected_prob]

num_trials = int(sys.argv[1])

baseline = 0
energy = []

# Parse data from CSV
with open('data_boltzmann.csv', newline = "\n") as data:
    reader = csv.reader(data, delimiter=",")
    for row in reader:
        baseline = float(row[0])
        energy = [float(_) for _ in row[1:-2]]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

energy_bar = []
for ebin in expected_energy:
    energy_bar.append(energy.count(ebin)/num_trials)

ax.bar(expected_energy, energy_bar, width = 8, label = "Actual distribution")
ax.bar(expected_energy, expected_prob, color = "orange", width = 2, label = "Expected probability distribution")

ax.legend()
ax.set_xlim([-800, 0])

plt.show()
