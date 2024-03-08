import torch
import copy
import time
import sys

from simulated_bifurcation import ConvergenceWarning, reset_env, set_env
from simulated_bifurcation.core import Ising
from simulated_bifurcation.optimizer import (
    SimulatedBifurcationEngine,
    SimulatedBifurcationOptimizer,
)

# Runs num_tests tests at each different coupling density.
# For each test, runs num_trials trials, and calculates the
# success probability, defined as the number of trials that
# achieve 95% as good of a hamiltonian as the reference.

# TODO: Add support for density sweeping

if (len(sys.argv) != 3):
    print("should be run as:") 
    print("sudo python3 test_64x64_density.py [num_tests] [num_trials]")
    sys.exit() 

num_tests = int(sys.argv[1])
num_trials = int(sys.argv[2])

#Run num_tests J matrixes.
for tests in range(num_tests):

    # Pick random integer couplings, and symmetrize.
    J = torch.randint(-7, 8, (63,63), dtype=torch.float32)
    J = torch.round((J + J.t()) / 2)
    h = torch.randint(-7, 8, (63,), dtype=torch.float32)
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
    print(sim_energy)

    # Run num_trials trials.
    for trials in range(num_trials):
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
            time_ms = 100
        )
        fpga_energy = ising.get_energy()
        print(fpga_energy)
        #if (fpga_energy <= 0.90*sim_energy):
        #    print(".", end="")
        #else:
        #    print("X", end="")
