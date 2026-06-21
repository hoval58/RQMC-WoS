import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dumbbell_array_rqmc import (
    wos_dumbbell_mc,
    wos_dumbbell_rqmc,
    wos_dumbbell_array_rqmc,
)


# Fixed parameters
N_list = 2 ** np.arange(2, 10)
n_rep = 100

# Dumbbell domain parameters
L = 1.5
R = 1.0
w = 0.4

# Starting point
x_0 = (0.5, 0.0)

epsilon = 1e-4
max_steps = 1000

# Where to save results
results_dir = Path("results_dumbbell_array")
results_dir.mkdir(exist_ok=True)


for N in N_list:
    N = int(N)

    mc_estimates = []
    qmc_sobol_estimates = []
    qmc_lattice_estimates = []
    qmc_sobol_estimates_array = []
    qmc_lattice_estimates_array = []




    print(f"\nRunning dumbbell experiments for N={N}")

    for i in tqdm(range(n_rep)):
        # Standard RQMC: Sobol
        res_sobol = wos_dumbbell_rqmc(
            x_0,
            w=w,
            L=L,
            R=R,
            N=N,
            type="sobol",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_sobol_estimates.append(res_sobol)

        # Standard RQMC: Lattice
        res_lattice = wos_dumbbell_rqmc(
            x_0,
            w=w,
            L=L,
            R=R,
            N=N,
            type="lattice",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_lattice_estimates.append(res_lattice)

        # Plain MC
        res_mc = wos_dumbbell_mc(
            x_0,
            w=w,
            L=L,
            R=R,
            N=N,
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        mc_estimates.append(res_mc)

        # Array-RQMC: Sobol
        res_sobol_array = wos_dumbbell_array_rqmc(
            x_0,
            w=w,
            L=L,
            R=R,
            N=N,
            qmc_type="sobol",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_sobol_estimates_array.append(res_sobol_array[0])


        # Array-RQMC: Korobov lattice
        res_lattice_array = wos_dumbbell_array_rqmc(
            x_0,
            w=w,
            L=L,
            R=R,
            N=N,
            qmc_type="lattice",
            max_steps=max_steps,
            seed=i,
            eps=epsilon,
        )
        qmc_lattice_estimates_array.append(res_lattice_array[0])
  

    results_dumbbell = {
        "N": N,
        "n_rep": n_rep,
        "x_0": x_0,
        "L": L,
        "R": R,
        "w": w,
        "epsilon": epsilon,
        "max_steps": max_steps,

        "mc_estimates": mc_estimates,
        "qmc_sobol_estimates": qmc_sobol_estimates,
        "qmc_lattice_estimates": qmc_lattice_estimates,
        "qmc_sobol_estimates_array": qmc_sobol_estimates_array,
        "qmc_lattice_estimates_array": qmc_lattice_estimates_array
    }

    output_file = results_dir / f"N={N}.pickle"

    with output_file.open("wb") as file:
        pickle.dump(results_dumbbell, file)

    print(f"Saved results to {output_file}")
