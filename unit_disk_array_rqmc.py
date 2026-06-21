import numpy as np
import random
import math
from tqdm import tqdm
import qmcpy
from qmcpy import DigitalNet
from qmcpy import Sobol
from qmcpy import Halton
import json
from pathlib import Path
from collections import Counter

import hilbertcurve
from hilbertcurve.hilbertcurve import HilbertCurve


# LatNet Builder generators
GENERATOR_FILE_DIM2 = Path(__file__).resolve().parent / "korobov_generators_dim2.json"

with GENERATOR_FILE_DIM2.open("r") as file:
    KOROBOV_DIM2 = json.load(file)

def get_korobov_a_dim2(n):
    try:
        a = int(KOROBOV_DIM2[str(n)]["a"])
    except KeyError as exc:
        raise KeyError(
            f"No dimension-2 Korobov generator stored for N={n} "
            f"in {GENERATOR_FILE_DIM2}."
        ) from exc

 
    if n == 4 and a == 1:    # The paper assumes a > 1
        a = 3

    if not (1 < a < n) or math.gcd(a, n) != 1:
        raise RuntimeError(
            f"Invalid stored Korobov generator a={a} for N={n}."
        )

    return a


#### HELPER FUNCTIONS

def boundary_value(x):
    """Boundary function g(x1, x2)."""
    return 0.5 * np.log((x[0]-2)**2 + x[1]**2)


def wos_unit_disk_mc(x0, N=100, max_steps=1000, eps=1e-5,seed=None):

    rng_seed = None if seed is None else seed + 2025
    rng = np.random.default_rng(rng_seed)
    points = rng.uniform(0, 2*np.pi, size=(N, max_steps))

    results = []
    walk_length = []
    for i in range(N):
        x = np.array(x0, dtype=float)
        step = 0

        while step < max_steps:
            r = 1 - np.linalg.norm(x)
            if r <= eps:
                walk_length.append(step)
                break
            theta = points[i, step]
            x += r * np.array([np.cos(theta), np.sin(theta)])
            step += 1

        x_boundary = x / np.linalg.norm(x)
        results.append(boundary_value(x_boundary))

    return (np.mean(results), np.mean(walk_length))

def wos_unit_disk_rqmc(x0, N, type, max_steps=1000, eps=1e-5, seed=None):
    """
    Walk-on-Spheres in the unit disk using RQMC for N trajectories.

    Parameters:
        x0: starting point 
        N: number of trajectories
        max_steps: maximum allowed WOS steps
        eps: termination threshold
        seed: scrambling seed
    Returns:
        WOS estimator from N trajectories, average number of steps needed
    """
    x0 = np.array(x0, dtype=float)
    results = []
    walk_length=[]
    if type=="sobol":
      sobol = Sobol(dimension=max_steps, seed=seed, randomize=True)
      points = sobol.gen_samples(N)  
    elif type=="halton":
      halton=Halton(dimension=max_steps, seed=seed, randomize=True)
      points=halton.gen_samples(N)
    elif type=="niederreiter":
      nied = qmcpy.DigitalNet(dimension=max_steps, generating_matrices="ebert_osisiogu.niederreiter_32.txt",randomize="LMS_DS",seed=seed)
      points = nied.gen_samples(n_min=0, n_max=N)
    elif type=="lattice":
      lattice = qmcpy.Lattice(dimension=max_steps, randomize='SHIFT',seed=seed)
      points = lattice.gen_samples(n=N)
    else:
        raise ValueError(f"Unknown qmc_type: {qmc_type}")


    for i in range(N):
        x = x0.copy()
        cpt=0
        for u in points[i]:
            r = 1 - np.linalg.norm(x)
            if r <= eps:
                walk_length.append(cpt)
                break
            cpt+=1
            theta = 2 * np.pi * u
            x += r * np.array([np.cos(theta), np.sin(theta)])
        x_boundary = x / np.linalg.norm(x)
        results.append(boundary_value(x_boundary))

    return (np.mean(results),np.mean(walk_length))

def disk_to_unit_square_xy(xy: np.ndarray) -> np.ndarray:
    """
    xy: (n,2) array with points in unit disk 
    returns uv in [0,1)^2
    """
    uv = 0.5 * (xy + 1.0)             # maps [-1,1] -> [0,1]
    eps = np.nextafter(1.0, 0.0)      
    return np.clip(uv, 0.0, eps)

p = 13
hc = HilbertCurve(p, 2)
nside = 1 << p

def hilbert_keys_hilbertcurve(xy: np.ndarray) -> np.ndarray:
    uv = disk_to_unit_square_xy(xy)
    ij = np.floor(uv * nside).astype(int)
    ij = np.clip(ij, 0, nside - 1)
    return np.array([hc.distance_from_point([int(ix), int(iy)]) for ix, iy in ij])


def lattice_korobov(n, a, seed=None):
    """
    Generate the shifted transition coordinate for the 2D Korobov rule

        ((i + 1/2)/n, (a*i/n + Delta) mod 1),

    where the first coordinate is implicit through the Hilbert rank.
    """
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer at least 4.")

    if n & (n - 1):
        raise ValueError("n must be a power of two.")

    if not (1 < a < n) or math.gcd(a, n) != 1:
        raise ValueError(
            f"Invalid Korobov generator a={a} for n={n}."
        )

    rng = np.random.default_rng(seed)
    delta = rng.random()

    i = np.arange(n, dtype=np.int64)
    u = (((a * i) % n) / n + delta) % 1.0

    return u



def wos_unit_disk_array_rqmc(x0, N, qmc_type="sobol", max_steps=1000, eps=1e-5, seed=None):
    """
    Array-RQMC Walk-on-Spheres in the unit disk.

    At each step:
      - generate N 1D RQMC uniforms
      - sort walkers by Hilbert key (inactive walkers pushed to the end)
      - assign u by rank (k-th in sorted order gets u[k])
      - move only active walkers 
    """
    x0 = np.array(x0, dtype=float)

    walkers = np.tile(x0, (N, 1))        
    active = np.ones(N, dtype=bool)
    walk_lengths = np.zeros(N, dtype=int)

    
    korobov_a = None
    if qmc_type == "lattice":
        korobov_a = get_korobov_a_dim2(N)

    for step in range(max_steps):
        if not np.any(active):
            break

        # --- Fresh RQMC point set at each step ---
        step_seed = None if seed is None else seed * (max_steps + 1) + step

        if qmc_type == "sobol":
            sampler = Sobol(dimension=1, seed=step_seed, randomize=True)
            u = sampler.gen_samples(N).reshape(-1)   
        elif qmc_type == "lattice":
            u = lattice_korobov(
                n=N,
                a=korobov_a,
                seed=step_seed,
            )
        else:
            raise ValueError(f"Unknown qmc_type: {qmc_type}")

        # --- Sort all walkers by Hilbert key; inactive walkers go last ---
        keys = hilbert_keys_hilbertcurve(walkers)           
        keys = np.where(active, keys, np.inf)               # push inactive to end
        order = np.argsort(keys, kind="mergesort")          

        # --- Assign u by rank in sorted order ---
        assigned_u = np.empty(N)
        assigned_u[order] = u   # walker at position order[k] gets u[k]

        # --- Move active walkers only ---
        r = 1.0 - np.linalg.norm(walkers, axis=1)
        to_move = active & (r > eps)

        theta = 2.0 * np.pi * assigned_u
        walkers[to_move] += r[to_move, None] * np.stack(
            [np.cos(theta[to_move]), np.sin(theta[to_move])], axis=1
        )

        # --- Update active flags and walk lengths ---
        r_new = 1.0 - np.linalg.norm(walkers, axis=1)
        newly_done = active & (r_new <= eps)
        walk_lengths[newly_done] = step + 1
        active[newly_done] = False

    # --- Evaluate boundary value for all walkers ---
    norms = np.linalg.norm(walkers, axis=1, keepdims=True)
    x_boundary = walkers / np.clip(norms, 1e-12, None)
    results = np.array([boundary_value(x_boundary[i]) for i in range(N)])

    return np.mean(results), np.mean(walk_lengths)



