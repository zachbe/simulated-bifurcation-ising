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
    print("sudo python3 test_gaussian.py [num_samples]")
    sys.exit() 

num_trials = int(sys.argv[1])

f = open("data_gaussian_rand.csv", "w+")

# Pick known-good couplings
J = torch.tensor([[-4,-1,-2],[-1,-4,0],[-2,0,-4]]).float()
h = torch.tensor([-0, -0, -0]).float()

print(-J)
print(np.linalg.inv(-J))
print(-h.numpy())
print(np.linalg.inv(-J) @ -h.numpy())

with open("gaussian_J_rand.npy", "wb+") as bf:
    np.save(bf, J.numpy())

with open("gaussian_h_rand.npy", "wb+") as bf:
    np.save(bf, h.numpy())

# Convert to sparse representation
new_J = [[] for _ in range(len(J))]
for i in range(len(J)):
    for j in range(len(J)):
        if J[i][j] == -4:
            new_J[i].append(np.ones((20,20)))
        if J[i][j] == -2:
            new_J[i].append(np.kron(np.ones((10,10)), np.identity(2)))
        if J[i][j] == -1:
            new_J[i].append(np.kron(np.ones((5,5)), np.identity(4)))
        if J[i][j] == 0:
            new_J[i].append(np.zeros((20,20)))
        if j == len(J) - 1:
            new_J[i] = np.concatenate(new_J[i])

J = np.concatenate(new_J, axis=1)

print(J)

# h is not sparse for now
h = np.kron(h, np.ones(20))

# Extend
J = np.concatenate((J, np.zeros((3,60))), axis = 0)
J = np.concatenate((J, np.zeros((63,3))), axis = 1)
h = np.concatenate((h, np.zeros(3)), axis = 0)

J = torch.from_numpy(J).float()
h = torch.from_numpy(h).float()

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
        autoscale = False,
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

with open("gaussian_samples_rand.npy", "wb+") as bf:
    np.save(bf, data)
