import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pacman_array_rqmc import (
    wos_pacman_mc,
    wos_pacman_rqmc,
    wos_pacman_array_rqmc,
    true_solution,
)


# Fixed parameters
N_list = 2 ** np.arange(2, 10)
n_rep = 100

# Starting point
r0 = 0.1244
theta0 = -0.7906
x_0 = (r0 * np.cos(theta0), r0 * np.sin(theta0))

# Exact reference value from the known solution
ref_value = true_solution(r0, theta0)

epsilon = 1e-4
max_steps = 1000

# Where to save results
results_dir = Path("results_pacman_array")
results_dir.mkdir(exist_ok=True)


for N in N_list:
    N = int(N)

    mc_estimates = []
    qmc_sobol_estimates = []
    qmc_lattice_estimates = []
    qmc_sobol_estimates_array = []
    qmc_lattice_estimates_array = []

    walk_length_mc = []
    walk_length_sobol = []
    walk_length_lattice = []
    walk_length_sobol_array = []
    walk_length_lattice_array = []

    print(f"\nRunning pacman experiments for N={N}")

    for i in tqdm(range(n_rep)):
        # Standard RQMC: Sobol
        res_sobol = wos_pacman_rqmc(
            x_0,
            n_walks=N,
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
            type="sobol",
        )
        qmc_sobol_estimates.append(res_sobol[0])
        walk_length_sobol.append(res_sobol[1])

        # Standard RQMC: Lattice
        res_lattice = wos_pacman_rqmc(
            x_0,
            n_walks=N,
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
            type="lattice",
        )
        qmc_lattice_estimates.append(res_lattice[0])
        walk_length_lattice.append(res_lattice[1])

        # Plain MC
        res_mc = wos_pacman_mc(
            x_0,
            n_walks=N,
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        mc_estimates.append(res_mc[0])
        walk_length_mc.append(res_mc[1])

        # Array-RQMC: Sobol
        res_sobol_array = wos_pacman_array_rqmc(
            x_0,
            N=N,
            qmc_type="sobol",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_sobol_estimates_array.append(res_sobol_array[0])
        walk_length_sobol_array.append(res_sobol_array[1])

        # Array-RQMC: Lattice
        res_lattice_array = wos_pacman_array_rqmc(
            x_0,
            N=N,
            qmc_type="lattice",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_lattice_estimates_array.append(res_lattice_array[0])
        walk_length_lattice_array.append(res_lattice_array[1])

    results_pacman = {
        "N": N,
        "n_rep": n_rep,
        "r0": r0,
        "theta0": theta0,
        "x_0": x_0,
        "ref_value": ref_value,
        "epsilon": epsilon,
        "max_steps": max_steps,

        "mc_estimates": mc_estimates,
        "qmc_sobol_estimates": qmc_sobol_estimates,
        "qmc_lattice_estimates": qmc_lattice_estimates,
        "qmc_sobol_estimates_array": qmc_sobol_estimates_array,
        "qmc_lattice_estimates_array": qmc_lattice_estimates_array,

        "walk_length_mc": walk_length_mc,
        "walk_length_sobol": walk_length_sobol,
        "walk_length_lattice": walk_length_lattice,
        "walk_length_sobol_array": walk_length_sobol_array,
        "walk_length_lattice_array": walk_length_lattice_array,
    }

    output_file = results_dir / f"N={N}.pickle"

    with output_file.open("wb") as file:
        pickle.dump(results_pacman, file)

    print(f"Saved results to {output_file}")
