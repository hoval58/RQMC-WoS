import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from unit_sphere_array_rqmc import (
    boundary_value,
    wos_unit_sphere_mc,
    wos_unit_sphere_rqmc,
    wos_unit_sphere_array_rqmc,
)


# Fixed parameters
N_list = 2 ** np.arange(2, 10)
n_rep = 100

# Starting point
x_0 = (0.2, 0.3, -0.1)

# Exact value
ref_value = boundary_value(x_0)

epsilon = 1e-4
max_steps = 1000

# Where to save results
results_dir = Path("results_unit_sphere_array")
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

    print(f"\nRunning unit sphere experiments for N={N}")

    for i in tqdm(range(n_rep)):
        # Standard RQMC: Sobol
        res_sobol = wos_unit_sphere_rqmc(
            x_0,
            type="sobol",
            N=N,
            L=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_sobol_estimates.append(res_sobol[0])
        walk_length_sobol.append(res_sobol[1])

        # Standard RQMC: Lattice
        res_lattice = wos_unit_sphere_rqmc(
            x_0,
            type="lattice",
            N=N,
            L=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_lattice_estimates.append(res_lattice[0])
        walk_length_lattice.append(res_lattice[1])

        # Plain MC
        res_mc = wos_unit_sphere_mc(
            x_0,
            N=N,
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        mc_estimates.append(res_mc[0])
        walk_length_mc.append(res_mc[1])

        # Array-RQMC: Sobol
        res_sobol_array = wos_unit_sphere_array_rqmc(
            x_0,
            N=N,
            qmc_type="sobol",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_sobol_estimates_array.append(res_sobol_array[0])
        walk_length_sobol_array.append(res_sobol_array[1])

        # Array-RQMC: Korobov lattice.

        res_lattice_array = wos_unit_sphere_array_rqmc(
            x_0,
            N=N,
            qmc_type="lattice",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_lattice_estimates_array.append(res_lattice_array[0])
        walk_length_lattice_array.append(res_lattice_array[1])

    results_unit_sphere = {
        "N": N,
        "n_rep": n_rep,
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
        pickle.dump(results_unit_sphere, file)

    print(f"Saved results to {output_file}")
