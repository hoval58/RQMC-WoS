import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Same N values as in run_unit_disk_experiments.py
N_list = 2 ** np.arange(2, 10)

# Same starting point as in run_unit_disk_experiments.py
x_0 = (0.0, 0.5)

# Exact value of the solution at x_0
ref_value = 0.5 * np.log((x_0[0] - 2) ** 2 + x_0[1] ** 2)

results_dir = Path("results_unit_disk_array")
plots_dir = Path("plots_unit_disk")
plots_dir.mkdir(exist_ok=True)


def pooled_power_fit(y1, y2, start_idx=5):
    N_fit = np.tile(N_list[start_idx:], 2)
    y_fit = np.concatenate([
        np.asarray(y1[start_idx:]),
        np.asarray(y2[start_idx:])
    ])

    mask = np.isfinite(y_fit) & (y_fit > 0)

    # Fit log(y) = a + b log(N)
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

mse_mc_all = []
mse_sobol_all = []
mse_lattice_all = []
mse_sobol_array_all = []
mse_lattice_array_all = []


for N in N_list:
    filename = results_dir / f"N={int(N)}.pickle"

    with filename.open("rb") as file:
        results = pickle.load(file)

    mc_estimates = np.array(results["mc_estimates"])
    sobol_estimates = np.array(results["qmc_sobol_estimates"])
    lattice_estimates = np.array(results["qmc_lattice_estimates"])
    sobol_array_estimates = np.array(results["qmc_sobol_estimates_array"])
    lattice_array_estimates = np.array(results["qmc_lattice_estimates_array"])

    var_mc_all.append(np.var(mc_estimates))
    var_sobol_all.append(np.var(sobol_estimates))
    var_lattice_all.append(np.var(lattice_estimates))
    var_sobol_array_all.append(np.var(sobol_array_estimates))
    var_lattice_array_all.append(np.var(lattice_array_estimates))

    mse_mc_all.append(np.mean((mc_estimates - ref_value) ** 2))
    mse_sobol_all.append(np.mean((sobol_estimates - ref_value) ** 2))
    mse_lattice_all.append(np.mean((lattice_estimates - ref_value) ** 2))
    mse_sobol_array_all.append(np.mean((sobol_array_estimates - ref_value) ** 2))
    mse_lattice_array_all.append(np.mean((lattice_array_estimates - ref_value) ** 2))


# Convert to arrays
N_list = np.array(N_list)

var_mc_all = np.array(var_mc_all)
var_sobol_all = np.array(var_sobol_all)
var_lattice_all = np.array(var_lattice_all)
var_sobol_array_all = np.array(var_sobol_array_all)
var_lattice_array_all = np.array(var_lattice_array_all)

mse_mc_all = np.array(mse_mc_all)
mse_sobol_all = np.array(mse_sobol_all)
mse_lattice_all = np.array(mse_lattice_all)
mse_sobol_array_all = np.array(mse_sobol_array_all)
mse_lattice_array_all = np.array(mse_lattice_array_all)


# Variance plot
plt.figure(figsize=(7, 5))

plt.loglog(N_list, var_mc_all, "^-", label="MC")
plt.loglog(N_list, var_sobol_all, "s-", label="Sobol")
plt.loglog(N_list, var_lattice_all, "o-", label="Lattice")
plt.loglog(N_list, var_sobol_array_all, "s-", label="Array-RQMC Sobol")
plt.loglog(N_list, var_lattice_array_all, "o-", label="Array-RQMC lattice")

# Pooled reference slope using the two array-RQMC curves
pooled_power_fit(var_sobol_array_all, var_lattice_array_all, start_idx=5)

plt.xlabel("N")
plt.ylabel("Variance over replications")
plt.title("Unit disk WOS variance")
plt.legend()
plt.tight_layout()

variance_plot = plots_dir / "unit_disk_variance.png"
plt.savefig(variance_plot, dpi=300)
plt.show()


# MSE plot
plt.figure(figsize=(7, 5))

plt.loglog(N_list, mse_mc_all, "^-", label="MC")
plt.loglog(N_list, mse_sobol_all, "s-", label="Sobol")
plt.loglog(N_list, mse_lattice_all, "o-", label="Lattice")
plt.loglog(N_list, mse_sobol_array_all, "s-", label="Array-RQMC Sobol")
plt.loglog(N_list, mse_lattice_array_all, "o-", label="Array-RQMC lattice")

# Pooled reference slope using the two array-RQMC MSE curves
pooled_power_fit(mse_sobol_array_all, mse_lattice_array_all, start_idx=5)

plt.xlabel("N")
plt.ylabel("MSE")
plt.title("Unit disk WOS MSE")
plt.legend()
plt.tight_layout()

mse_plot = plots_dir / "unit_disk_mse.png"
plt.savefig(mse_plot, dpi=300)
plt.show()


print(f"Reference value: {ref_value}")
print(f"Saved variance plot to {variance_plot}")
print(f"Saved MSE plot to {mse_plot}")
