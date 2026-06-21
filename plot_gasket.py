import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_list = 2 ** np.arange(2, 10) 


plots_dir = Path("plots_gasket")
plots_dir.mkdir(exist_ok=True)


def pooled_power_fit(y1, y2, start_idx=5): # for the pooled reference line
    N_fit = np.tile(N_list[start_idx:], 2)
    y_fit = np.concatenate([
        np.asarray(y1[start_idx:]),
        np.asarray(y2[start_idx:])
    ])

    mask = np.isfinite(y_fit) & (y_fit > 0)


    b, a = np.polyfit(np.log(N_fit[mask]), np.log(y_fit[mask]), 1)

    C = np.exp(a)
    ref_line = C * N_list ** b

    plt.plot(
        N_list,
        ref_line,
        "--",
        label=f"Slope = {b:.2f}"
    )

    return b, C



var_mc_all = []
var_sobol_all = []
var_lattice_all = []
var_sobol_array_all = []
var_lattice_array_all = []




for N in N_list:
    with open(f"results_gasket_array/N={int(N)}.pickle", "rb") as file:
        results = pickle.load(file)

    mc_estimates = np.array(results["mc_estimates"])
    sobol_estimates = np.array(results["qmc_sobol_estimates"])
    lattice_estimates = np.array(results["qmc_lattice_estimates"])


    sobol_array_estimates = np.array(results["qmc_sobol_estimates_array"])
    lattice_array_estimates = np.array(results["qmc_lattice_estimates_array"])

    # MC and Plain RQMC
    var_mc_all.append(np.var(mc_estimates))
    var_sobol_all.append(np.var(sobol_estimates))
    var_lattice_all.append(np.var(lattice_estimates))

    # Array RQMC
    var_sobol_array_all.append(np.var(sobol_array_estimates))
    var_lattice_array_all.append(np.var(lattice_array_estimates))




# Convert to arrays
N_list = np.array(N_list)

var_mc_all = np.array(var_mc_all)
var_sobol_all = np.array(var_sobol_all)
var_lattice_all = np.array(var_lattice_all)
var_sobol_array_all = np.array(var_sobol_array_all)
var_lattice_array_all = np.array(var_lattice_array_all)




# Plot variances
plt.figure(figsize=(7, 5))

plt.plot(N_list, var_mc_all, marker="^", linestyle="-", label="MC")
plt.plot(N_list, var_sobol_all, marker="s", linestyle="-", label="Sobol",markerfacecolor='none')
plt.plot(N_list, var_lattice_all, marker="o", linestyle="-", label="Lattice", markerfacecolor='none')
plt.plot(N_list, var_sobol_array_all, marker="s", linestyle="-", label="Array-Sobol")
plt.plot(N_list, var_lattice_array_all, marker="o", linestyle="-", label="Array-Lattice")

# Pooled reference slope using the two array-RQMC variance curves
pooled_power_fit(var_sobol_all, var_lattice_all, start_idx=5)
pooled_power_fit(var_sobol_array_all, var_lattice_array_all, start_idx=5)

plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of trajectories")
plt.ylabel("Variance")
plt.tight_layout()

variance_plot = plots_dir / "gasket_variance.png"
plt.savefig(variance_plot, dpi=300)
plt.show()


