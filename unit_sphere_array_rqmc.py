import numpy as np
import random
import math
from tqdm import tqdm
from scipy.stats import qmc,norm,uniform
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
GENERATOR_FILE_DIM3 = Path(__file__).resolve().parent / "korobov_generators_dim3.json"

with GENERATOR_FILE_DIM3.open("r") as file:
    KOROBOV_DIM3 = json.load(file)



def get_korobov_generator_dim3(n):
    """
    Return (a, generating_vector) for the 3D Korobov rule

        (1, a, a^2) mod n.
    """
    n = int(n)

    try:
        entry = KOROBOV_DIM3[str(n)]
    except KeyError as exc:
        raise KeyError(
            f"No dimension-3 Korobov generator stored for N={n}."
        ) from exc

    a = int(entry["a"])
    stored_vector = np.asarray(
        entry["generating_vector"],
        dtype=np.int64,
    )

    # Replace the trivial N=4 solution if LatNet Builder returns a=1.
    if n == 4 and a == 1:
        a = 3
        generating_vector = np.array(
            [pow(a, j, n) for j in range(3)],
            dtype=np.int64,
        )
    else:
        if not (1 < a < n):
            raise RuntimeError(
                f"Invalid Korobov multiplier a={a} for N={n}; "
                "we require 1 < a < N."
            )

        if math.gcd(a, n) != 1:
            raise RuntimeError(
                f"Korobov multiplier a={a} is not coprime to N={n}."
            )

        expected_vector = np.array(
            [pow(a, j, n) for j in range(3)],
            dtype=np.int64,
        )

        if stored_vector.shape != (3,):
            raise RuntimeError(
                f"Stored generating vector must have length 3; "
                f"got {stored_vector.tolist()}."
            )

        if not np.array_equal(stored_vector, expected_vector):
            raise RuntimeError(
                "Stored generating vector is inconsistent with a.\n"
                f"Stored:   {stored_vector.tolist()}\n"
                f"Expected: {expected_vector.tolist()}"
            )

        generating_vector = stored_vector

    return a, generating_vector


def lattice_korobov_3d(n, a, generating_vector, seed=None):
    """
    Generate the two transition coordinates from the 3D Korobov rule

        (1, a, a^2) mod n.

    The first coordinate is implicit through the Array-RQMC rank.
    """
    n = int(n)
    a = int(a)
    generating_vector = np.asarray(
        generating_vector,
        dtype=np.int64,
    )

    if generating_vector.shape != (3,):
        raise ValueError(
            "generating_vector must have shape (3,)."
        )

    expected_vector = np.array(
        [pow(a, j, n) for j in range(3)],
        dtype=np.int64,
    )

    if not np.array_equal(generating_vector, expected_vector):
        raise ValueError(
            "The generating vector is inconsistent with a.\n"
            f"Received: {generating_vector.tolist()}\n"
            f"Expected: {expected_vector.tolist()}"
        )

    rng = np.random.default_rng(seed)

    # Independent random shifts for the two transition coordinates.
    shifts = rng.random(2)

    i = np.arange(n, dtype=np.int64)[:, None]
    transition_generators = generating_vector[1:]

    U = (
        (i * transition_generators[None, :]) % n
    ) / n

    U = (U + shifts[None, :]) % 1.0

    return U


#### HELPER FUNCTIONS

def boundary_value(x):
    """Boundary function g(x1, x2, x3)."""
    return ((x[0]-2)**2 + x[1]**2 + x[2]**2)**(-0.5)


def wos_unit_sphere_mc(x0, N=100, max_steps=1000, eps=1e-5, seed=None):

    rng = np.random.default_rng(seed)
    phi_list = rng.uniform(0, 2*np.pi, size=(N, max_steps))
    cos_samples = rng.uniform(-1, 1, size=(N, max_steps))
    sin_samples=np.sqrt(1-cos_samples**2)
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
            phi = phi_list[i, step]
            cos_theta = cos_samples[i, step]
            sin_theta = sin_samples[i, step]

            u = np.array([sin_theta * np.cos(phi),
              sin_theta * np.sin(phi),
              cos_theta])
            x += r * u
            step += 1

        x_boundary = x / np.linalg.norm(x)
        results.append(boundary_value(x_boundary))

    return (np.mean(results), np.mean(walk_length))


def wos_unit_sphere_rqmc(x0, type, N=2**12, L=150, eps=1e-5, seed=1234):
    """
    Walk-on-Spheres with plain RQMC in unit sphere
    """
    x0 = np.asarray(x0, dtype=float)
    assert np.linalg.norm(x0) < 1.0, "Start inside the unit ball."

    estimates = []
    wl_all = []


    if type=="sobol":
        sob = qmc.Sobol(d=2*L, scramble=True, seed=seed)
        U = sob.random(N)                        


    elif type=="halton":
      halton=qmc.Halton(d=2*L, scramble=True, seed=seed)
      U = halton.random(N)
    elif type=="niederreiter":
      nied = qmcpy.DigitalNet(dimension=2*L, generating_matrices="ebert_osisiogu.niederreiter_32.txt",randomize="LMS_DS",seed=seed)
      U= nied.gen_samples(n_min=0, n_max=N)
    elif type=="lattice":
      lattice = qmcpy.Lattice(dimension=2*L, randomize='SHIFT',seed=seed)
      U = lattice.gen_samples(n=N)
    else:
        raise ValueError(f"Unknown type: {type}")


    U = np.clip(U, 1e-12, 1-1e-12)

    vals = np.empty(N)
    wlen = np.empty(N, dtype=int)

    for i in range(N):
        x = x0.copy()
        step = 0
            
        for j in range(L):
            r_ball = 1.0 - np.linalg.norm(x)
            if r_ball <= eps:
                    
                step = j
                break

            u1 = U[i, 2*j]
            u2 = U[i, 2*j+1]
            phi = 2.0*np.pi*u1
            cos_theta = 2.0*u2 - 1.0
            sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta*cos_theta))

            dir_vec = np.array([sin_theta*np.cos(phi),
                                    sin_theta*np.sin(phi),
                                    cos_theta])
            x += r_ball * dir_vec

            
        if np.linalg.norm(x) < 1.0:
            x = x / np.linalg.norm(x)

        x_bdry = x / np.linalg.norm(x)
        vals[i] = boundary_value(x_bdry)
        wlen[i] = step

    estimates.append(vals.mean())
    wl_all.append(wlen.mean())

    return (float(np.mean(estimates)), float(np.mean(wl_all)))



p = 13
hc = HilbertCurve(p, 3)     
nside = 1 << p

def disk_to_unit_xy(xyz: np.ndarray) -> np.ndarray:
    """
    xyz: (n,3) array with points in unit disk (x^2+y^2+z^2<= 1)
    returns uv in [0,1)^3
    """
    uvw = 0.5 * (xyz + 1.0)             # maps [-1,1] -> [0,1]
    eps = np.nextafter(1.0, 0.0)     
    return np.clip(uvw, 0.0, eps)

def hilbert_keys_hilbertcurve(xyz: np.ndarray) -> np.ndarray:
    uvw = disk_to_unit_xy(xyz)
    ijk = np.floor(uvw * nside).astype(int)
    ijk = np.clip(ijk, 0, nside - 1)
    return np.array([hc.distance_from_point([int(ix), int(iy), int(iz)]) for ix, iy, iz in ijk])


def wos_unit_sphere_array_rqmc(x0, N, qmc_type="sobol", max_steps=1000, eps=1e-5, seed=None):
    """
    Array-RQMC Walk-on-Spheres in the unit sphere
    """
    x0 = np.array(x0, dtype=float)

    walkers = np.tile(x0, (N, 1)) 
    active = np.ones(N, dtype=bool)
    walk_lengths = np.full(N, max_steps, dtype=int)

    korobov_a = None
    korobov_vector = None

    if qmc_type == "lattice":
        korobov_a, korobov_vector = get_korobov_generator_dim3(N)

    for step in range(max_steps):
        if not np.any(active):
            break

        step_seed = None if seed is None else seed * (max_steps + 1) + step

        # --- Fresh RQMC: N points in 2D (two uniforms per step) ---
        if qmc_type == "sobol":
            sampler = Sobol(dimension=2, seed=step_seed, randomize=True)
            U = sampler.gen_samples(N)          # (N, 2)
        elif qmc_type == "halton":
            sampler = Halton(dimension=2, seed=step_seed, randomize=True)
            U = sampler.gen_samples(N)
        elif qmc_type == "lattice":
            U = lattice_korobov_3d(
                n=N,
                a=korobov_a,
                generating_vector=korobov_vector,
                seed=step_seed,
            )
        else:
            raise ValueError(f"Unknown qmc_type: {qmc_type}")

        u1 = U[:, 0]   # for phi
        u2 = U[:, 1]   # for cos(theta)

        # --- Sort ALL walkers by key; inactive pushed to end (Approach A) ---
        keys = hilbert_keys_hilbertcurve(walkers)     # (N,)
        keys = np.where(active, keys, np.inf)
        order = np.argsort(keys, kind="mergesort")    # stable sort for ties

        # Assign by rank: k-th in sorted order gets (u1[k], u2[k])
        assigned_u1 = np.empty(N)
        assigned_u2 = np.empty(N)
        assigned_u1[order] = u1
        assigned_u2[order] = u2

        # --- Move active walkers only ---
        r = 1.0 - np.linalg.norm(walkers, axis=1)     # (N,)
        to_move = active & (r > eps)

        phi = 2.0 * np.pi * assigned_u1
        cos_theta = 2.0 * assigned_u2 - 1.0
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))

        dir_vec = np.stack(
            [sin_theta * np.cos(phi),
             sin_theta * np.sin(phi),
             cos_theta],
            axis=1
        )  # (N, 3)

        walkers[to_move] += r[to_move, None] * dir_vec[to_move]

        # --- Absorb those within eps of boundary (and record length) ---
        r_new = 1.0 - np.linalg.norm(walkers, axis=1)
        newly_done = active & (r_new <= eps)
        walk_lengths[newly_done] = step + 1
        active[newly_done] = False

    # --- Evaluate boundary value on the boundary ---
    norms = np.linalg.norm(walkers, axis=1, keepdims=True)
    x_boundary = walkers / np.clip(norms, 1e-12, None)
    results = np.array([boundary_value(x_boundary[i]) for i in range(N)])

    return np.mean(results), np.mean(walk_lengths)





