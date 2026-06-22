# Randomized Quasi-Monte Carlo for Walk-on-Spheres 

This repository contains Python code for experiments with Walk-on-Spheres (WoS), randomized quasi-Monte Carlo (RQMC), and Array-RQMC methods.
The code compares standard Monte Carlo, standard RQMC and Array-RQMC on several examples involving two- and three-dimensional domains presented in our two papers ["Randomized Quasi-Monte Carlo for Walk on Spheres"](https://arxiv.org/abs/2605.12844) and ["Walk on Spheres and Array-RQMC"](https://arxiv.org/abs/2605.12844). 

It also contains the code used to compute the column-wise total Sobol indices for the gasket example (`gasket_sobol_indices.py`).


## Contents
The repository includes simulation, experiment-running, plotting, and Sobol indices computation. 

Main domain scripts include:
`gasket_array_rqmc.py`
`unit_disk_array_rqmc.py`
`dumbbell_array_rqmc.py`
`pacman_array_rqmc.py`
`unit_sphere_array_rqmc.py`

Experiment scripts include files such as:
`run_gasket_experiments.py`
`run_unit_disk_experiments.py`
`run_dumbbell_experiments.py`
`run_pacman_experiments.py`
`run_unit_sphere_experiments.py`

The experiments in these files are run for sample sizes ranging from $4$ to $2^9$, but the user can extend the range of sample sizes by modifying `N_list`.


Plotting scripts include files such as:
`plot_gasket.py`
`plot_unit_disk.py`
`plot_dumbbell.py`
`plot_pacman.py`
`plot_unit_sphere.py`

The Sobol-index diagnostic script is:
`gasket_sobol_indices.py`
This script computes column-wise total Sobol indices for the gasket WoS estimator under four methods:
`mc`: standard Monte Carlo, no Hilbert sorting;
`rqmc`: standard RQMC, no Hilbert sorting;
`array_mc`: Array-MC with Hilbert sorting;
`array_rqmc`: Array-RQMC with Hilbert sorting.

## Installation
Create and activate a Python environment. For example, with conda:
```bash
conda create -n rqmc python=3.11
conda activate rqmc
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
The required packages are listed in `requirements.txt`. In particular, the code requires packages such as `qmcpy` and `hilbertcurve`.

## Korobov Lattice generator files
Some scripts use precomputed LatNet Builder Korobov lattice generators stored as JSON files.

To run Array-RQMC for the gasket, dumbbell and unit disk examples, place `korobov_generators_dim2.json` in the same folder as the running scripts.

For the unit sphere and pacman examples, place `korobov_generators_dim3.json` and `korobov_generators_dim4.json`, respectively, in the same folder as the corresponding script.

These JSON files are expected to contain entries indexed by sample size `N`, with fields such as `a` and `generating_vector`.

## Running the main experiments
From the repository folder, run commands such as:
```bash
python run_unit_disk_experiments.py
python run_dumbbell_experiments.py
python run_pacman_experiments.py
python run_unit_sphere_experiments.py
```
These scripts generate `.pickle` result files in folders such as:
```text
results_unit_disk_array/
results_dumbbell_array/
results_pacman_array/
results_unit_sphere_array/
```

## Plotting results
After running an experiment, use the corresponding plotting script. For example:
```bash
python plot_unit_disk.py
python plot_pacman.py
python plot_unit_sphere.py
```
The plot scripts save figures in folders such as:
```text
plots_unit_disk/
plots_pacman/
plots_unit_sphere/
```

## Column-wise total Sobol indices
To compute the column-wise total Sobol indices for the gasket example, use:
```bash
python gasket_sobol_indices.py --method mc
```
```bash
python gasket_sobol_indices.py --method rqmc --qmc_type lattice
```
```bash
python gasket_sobol_indices.py --method array_mc
```
```bash
python gasket_sobol_indices.py --method array_rqmc --qmc_type lattice
```
For `method=array_rqmc`, the option `qmc_type=lattice` uses the precomputed lattice stored in `korobov_generators_dim2.json`. So, the json file will need to be included in the same folder as the running script.

For `method=rqmc`, the option `qmc_type=lattice` uses the lattice implementation from QMCPy.

The output is saved by default in:
```text
results_gasket_sobol_indices/
```



