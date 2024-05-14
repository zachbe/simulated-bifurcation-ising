
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# Data should contain num_tests tests at each different coupling density.
# For each test, there will have been num_trials trials.

if (len(sys.argv) != 2):
    print("should be run as:") 
    print("sudo python3 plot_data_boltzmann.py [num_samples] [num_trials]")
    sys.exit() 

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

ax.hist(energy, 15)

plt.show()
