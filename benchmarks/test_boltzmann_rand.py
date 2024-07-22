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

f = open("data_boltzmann_rand.csv", "w+")

# Pick random integer couplings
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

f.write(str(sim_energy[0].item())+",")

data = np.zeros((num_trials, 63))
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
        reprogram_J = (trial == 0),
        counter_cutoff = 0,
        counter_max = 1
    )
    fpga_energy = ising.get_energy()
    print(f"{trial}")
    snp = ising.computed_spins.numpy()
    for i in range(len(snp)):
        data[trial][i] = snp[i][0]
    f.write(str(fpga_energy[0].item())+",")

with open("boltzmann_J_rand.npy", "wb+") as bf:
    np.save(bf, J.numpy())
with open("boltzmann_h_rand.npy", "wb+") as bf:
    np.save(bf, h.numpy())
with open("boltzmann_samples_rand.npy", "wb+") as bf:
    np.save(bf, data)