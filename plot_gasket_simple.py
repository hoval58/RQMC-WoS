import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Same N range as in the experiments
N_list = 2 ** np.arange(2, 10)


# Lists for variances
var_mc_all = []
var_sobol_all = []
var_lattice_all = []
var_sobol_array_all = []
var_lattice_array_all = []

# Optional: MSE lists, computed as variance + squared bias
mse_sobol_array_all = []
mse_lattice_array_all = []

# Read the pickle files
for N in N_list:
    with open(f"results_gasket_array/N={int(N)}.pickle", "rb") as file:
        results = pickle.load(file)

    # Standard methods
    var_mc_all.append(np.var(results["mc_estimates"]))
    var_sobol_all.append(np.var(results["qmc_sobol_estimates"]))
    var_lattice_all.append(np.var(results["qmc_lattice_estimates"]))

    # Array methods
    # If your file uses the names qmc_sobol_estimates_array_fresh and
    # qmc_lattice_estimates_array_fresh instead, replace the two keys below.
    var_sobol_array_all.append(np.var(results["qmc_sobol_estimates_array"]))
    var_lattice_array_all.append(np.var(results["qmc_lattice_estimates_array"]))



# Plot variances
plt.figure(figsize=(7, 5))

plt.plot(N_list, var_mc_all, marker="^", linestyle="-", label="MC")
plt.plot(N_list, var_sobol_all, marker="s", linestyle="-", label="Sobol")
plt.plot(N_list, var_lattice_all, marker="o", linestyle="-", label="Lattice")
plt.plot(N_list, var_sobol_array_all, marker="s", linestyle="-", label="Array-Sobol")
plt.plot(N_list, var_lattice_array_all, marker="o", linestyle="-", label="Array-Lattice")

# Reference slope line
C = var_sobol_all[-1] * N_list[-1] ** 1.15
ref_line = C * np.array(N_list) ** (-1.15)
plt.plot(N_list, ref_line, "--", label="Slope -1.15")

plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of trajectories")
plt.ylabel("Variance")
plt.tight_layout()

Path("plots_gasket").mkdir(exist_ok=True)
plt.savefig("plots_gasket/gasket_variance.png", dpi=300)
plt.show()


