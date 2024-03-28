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

# For each test, runs num_trials trials, and calculates the
# success probability, defined as the number of trials that
# achieve 95% as good of a hamiltonian as the reference.

if (len(sys.argv) != 3):
    print("should be run as:") 
    print("sudo python3 test_64x64_noise.py [num_tests] [num_trials]")
    sys.exit() 

num_tests = int(sys.argv[1])
num_trials = int(sys.argv[2])

f = open("data_noise.csv", "w+")

# TODO: This is a workaround to an issue where repeatedly re-initializing
# the FPGA occasionally causes errors.
J = torch.randint(-7, 8, (63,63), dtype=torch.float32)
h = torch.randint(-7, 8, (63,), dtype=torch.float32)
ising = Ising(J, h, use_fpga = True, digital_ising_size=64)

#Run num_tests J matrixes.
for tests in range(num_tests):

    # Pick random integer couplings.
    J = torch.randint(-7, 8, (63,63), dtype=torch.float32)
    J = torch.round((J + J.t()) / 2)
    h = torch.randint(-7, 8, (63,), dtype=torch.float32)
    ising.J = J
    ising.h = h

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

    # Run num_trials trials.
    # We can't use agents here, because we need to
    # re-shuffle each time.
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
            cycles = 1000,
            shuffle_spins = False
        )
        fpga_energy = ising.get_energy()
        print(".")
        f.write(str(ising.time_elapsed)+",")
        f.write(str(fpga_energy[0].item())+",")

    f.write("\n")

