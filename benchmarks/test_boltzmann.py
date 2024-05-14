import torch
import copy
import time
import sys
import numpy as np

from simulated_bifurcation import ConvergenceWarning, reset_env, set_env
from simulated_bifurcation.core import Ising
from simulated_bifurcation.optimizer import (
    SimulatedBifurcationEngine,
    SimulatedBifurcationOptimizer,
)

if (len(sys.argv) != 2):
    print("should be run as:") 
    print("sudo python3 test_boltzmann.py [num_samples]")
    sys.exit() 

num_trials = int(sys.argv[1])

f = open("data_boltzmann.csv", "w+")

J = torch.zeros((63,63), dtype=torch.float32)
h = torch.full((63,),-7, dtype=torch.float32)

for i in range(0, 62):
    J[i][i+1] = -7
    J[i+1][i] = -7

ising = Ising(J, h, use_fpga = True, digital_ising_size=64)
    
# Use the simulated bifurcation algorithm to get a baseline solution.
ising.minimize(
    1,
    100000,
    False,
    False,
    False,
    use_window=False,
    sampling_period=50,
    convergence_threshold=50,
    use_fpga = False
)
expected_data = copy.deepcopy(ising.computed_spins)
sim_energy = ising.get_energy()

f.write(str(sim_energy[0].item())+",")

for trial in range(num_trials):
    ising.minimize(
        1,
        10000,
        False,
        False,
        False,
        use_window=False,
        sampling_period=50,
        convergence_threshold=50,
        use_fpga = True,
        cycles = 100000,
        shuffle_spins = False,
        reprogram_J = False,
        counter_cutoff = 0,
        counter_max = 1
    )
    fpga_energy = ising.get_energy()
    print(f"{trial}")
    f.write(str(fpga_energy[0].item())+",")
