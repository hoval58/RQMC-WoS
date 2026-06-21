import numpy as np
import random
import math
from tqdm import tqdm
from scipy.stats import qmc
import qmcpy
from qmcpy import DigitalNet
from qmcpy import Sobol
from qmcpy import Halton

import hilbertcurve
from hilbertcurve.hilbertcurve import HilbertCurve


def true_solution(r,theta):
    return(r**(1/3)*np.sin(theta/3)+np.exp(-0.5*r**2))
    
def distance_to_sector_boundary(point):
    """Return (R, which_boundary) where R is the Euclidean distance from 'point'
    to the boundary of the sector domain, and which_boundary in {1,2,3} indicates
    which boundary piece attains the minimum:
        1 -> circular arc r=1
        2 -> ray θ = 0 (x-axis, x>=0)
        3 -> ray θ = π/2 (y-axis, y>=0)
    """
    x, y = float(point[0]), float(point[1])
    r = np.hypot(x, y)

    # Distance to the circular arc r=1 (assuming inside domain, r<=1).
    d_circle = max(0.0, 1.0 - r)

    # Distance to ray at θ=0 (positive x-axis). For points with x>=0, it's |y|.
    # If x<0, the closest point on that ray is the origin, so distance is r.
    d_ray0 = abs(y) if x >= 0.0 else r

    # Distance to ray at θ=π/2 (positive y-axis). For y>=0, it's |x|.
    # If y<0, closest point on that ray is also the origin.
    d_ray90 = abs(x) if y >= 0.0 else r

    distances = [d_circle, d_ray0, d_ray90]
    which = int(np.argmin(distances)) + 1
    return distances[which - 1], which


def project_to_boundary(point):
    """
    Project 'point' to the closest boundary point of the sector and return
    (proj_point, which_boundary) with the same boundary labeling as above.
    """
    x, y = float(point[0]), float(point[1])
    r = np.hypot(x, y)
    R, which = distance_to_sector_boundary(point)

    if which == 1:
        # Project radially to the circle r=1 
        if r == 0.0:
            # Arbitrary angle; pick (1,0)
            proj = (1.0, 0.0)
        else:
            proj = (x / r, y / r)
    elif which == 2:
        # Project to theta=0 (x-axis, x>=0)
        if x >= 0.0:
            proj = (x, 0.0)
        else:
            proj = (0.0, 0.0)  # closest point on the ray is the origin
    else: 
        # Project to theta=pi/2 (y-axis, y>=0)
        if y >= 0.0:
            proj = (0.0, y)
        else:
            proj = (0.0, 0.0)

    return proj, which


# -------------------------------
# Sampling inside/on disks
# -------------------------------

def random_point_in_disk(center, R, rng):
    theta = rng.uniform(0.0, 2.0 * math.pi)
    rho = R * math.sqrt(rng.uniform(0.0, 1.0))
    return (center[0] + rho * math.cos(theta), center[1] + rho * math.sin(theta))


def random_point_on_circle(center, R, rng):
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return (center[0] + R * math.cos(theta), center[1] + R * math.sin(theta))



def f_rhs(point): #source function
    x, y = point
    r2 = x * x + y * y
    return -2.0 * (1.0 - 0.5 * r2) * np.exp(-0.5 * r2)


def G_disk(x, y, R):
    """Green's function for 2D Laplacian in a disk of radius R with Dirichlet BC,
    centered at x: G = 1/(2π) * log(|x - y| / R)."""
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    rho = np.hypot(dx, dy)
    rho = max(rho, 1e-15)
    return (1.0 / (2.0 * math.pi)) * np.log(rho / R) #negative sign compared to the paper


def g_boundary(point):
    """Evaluate Dirichlet boundary data at the closest boundary point.
    Returns g(proj(point)).

    NOTE: The boundary condition u(1,θ)=sin(θ/3)+e^{-1/2} is defined for θ∈[-3π/2,0].
    For points on the circular arc where atan2 returns θ>0 (i.e., θ∈(π/2,π]),
    we must map θ → θ-2π to land in [-3π/2,0] before applying sin(θ/3).
    """
    proj, which = project_to_boundary(point)
    x, y = proj
    r = math.hypot(x, y)
    theta = math.atan2(y, x)

    if which == 1:
        # arc r=1: map angle to the interval [-3π/2, 0]
        theta_bc = theta - 2.0 * math.pi if theta > 0.0 else theta
        return math.sin(theta_bc / 3.0) + math.exp(-0.5)
    elif which == 2:
        # ray θ=0 (x-axis): y=0, r=|x|
        return math.exp(-0.5 * r * r)
    else:
        # ray θ=π/2 (y-axis ≡ θ=-3π/2): x=0, r=|y|
        return - (r ** (1.0 / 3.0)) + math.exp(-0.5 * r * r)


# -------------------------------
# Walk-on-Spheres Monte Carlo solver
# -------------------------------

def wos_pacman_mc(x0, n_walks, max_steps=1000, eps=5e-5, seed=None):
    """
    Estimate u(x0) for the Poisson problem in the pacman domain
    using standard Monte Carlo Walk-on-Spheres.

    Parameters
    ----------
    x0 : Starting point.
    n_walks : Number of independent trajectories.
    max_steps : Maximum number of WOS steps per trajectory.
    eps : Termination threshold.
    seed : Random seed.

    Returns:
    
    mean_estimate : Average WOS estimator over the trajectories.
    mean_walk_length : Average number of WOS steps.
    """

    estimates = []
    walk_length = []

    rng = np.random.default_rng(seed)

    for _ in range(n_walks):
        x = (float(x0[0]), float(x0[1]))
        u_accum = 0.0
        cpt = 0

        for step in range(max_steps):
            R, _which = distance_to_sector_boundary(x)

            if R <= eps:
                break

            # Source term contribution
            y = random_point_in_disk(x, R, rng)
            u_accum += math.pi * (R ** 2) * G_disk(x, y, R) * f_rhs(y)

            # Move to a random point on the WOS circle
            x = random_point_on_circle(x, R, rng)
            cpt += 1

        walk_length.append(cpt)

        # Boundary term
        u_accum += g_boundary(x)
        estimates.append(u_accum)

    return float(np.mean(estimates)), float(np.mean(walk_length))


def random_point_in_disk_qmc(center, R, points_1, points_2, type=None):
    theta = 2*math.pi*points_1
    rho = R * np.sqrt(points_2)
    return (center[0] + rho * np.cos(theta), center[1] + rho * np.sin(theta))


def random_point_on_circle_qmc(center, R, points):
    theta = 2*math.pi*points
    return (center[0] + R * np.cos(theta), center[1] + R * np.sin(theta))



def wos_pacman_rqmc(
    x0,
    n_walks,
    max_steps,
    seed,
    eps = 5e-5,
    type = None
):
    """Estimate u(x0) for the Poisson problem using RQMC with Walk-on-Spheres."""

    estimates = []
    walk_length = []

    # total RQMC dimension: 3 * max_steps (u,v,w per step)
    dim = 3 * max_steps

    # --- Generate QMC samples ---
    if type == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        points = sampler.random(n_walks)

    elif type == "halton":
        sampler = qmc.Halton(d=dim, scramble=True, seed=seed)
        points = sampler.random(n_walks)

    elif type == "niederreiter":
        import qmcpy
        sampler = qmcpy.DigitalNet(
            dimension=dim,
            generating_matrices="ebert_osisiogu.niederreiter_32.txt",
            randomize="LMS_DS",
            seed=seed
        )
        points = sampler.gen_samples(n_min=0, n_max=n_walks)

    elif type == "lattice":
        import qmcpy
        sampler = qmcpy.Lattice(dimension=dim, randomize='SHIFT', seed=seed)
        points = sampler.gen_samples(n=n_walks)

    else:
        raise ValueError("Unknown QMC type. Choose from ['sobol', 'halton', 'niederreiter', 'lattice'].")

    
    for i in range(n_walks):
        x = (float(x0[0]), float(x0[1]))
        u_accum = 0.0
        step = 0

        # Iterate over triplets (u,v,w)
        for step in range(max_steps):
            u = points[i, 3 * step + 0]
            v = points[i, 3 * step + 1]
            w = points[i, 3 * step + 2]

            R, _which = distance_to_sector_boundary(x)
            if R <= eps:
                walk_length.append(step)
                break

            
            y = random_point_in_disk_qmc(x, R, u, v)
            y = (float(y[0]), float(y[1]))
            u_accum += math.pi * (R ** 2) * G_disk(x, y, R) * f_rhs(y) #source term contribution

            x = random_point_on_circle_qmc(x, R, w)
            x = (float(x[0]), float(x[1]))

        # Boundary term 
        u_accum += g_boundary((float(x[0]), float(x[1])))
        estimates.append(u_accum)

    return float(np.mean(estimates)), np.mean(walk_length)

## Array-RQMC

def disk_to_unit_square_xy(xy):
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

def hilbert_keys_hilbertcurve(xy):
    uv = disk_to_unit_square_xy(xy)
    ij = np.floor(uv * nside).astype(int)
    ij = np.clip(ij, 0, nside - 1)
    return np.array([hc.distance_from_point([int(ix), int(iy)]) for ix, iy in ij])


    
def wos_pacman_array_rqmc(x0, N, qmc_type="sobol", max_steps=1000, eps=1e-5, seed=None):
    """
    Array-RQMC WOS for the pacman domain

    """
    x0 = np.array(x0, dtype=float)

    walkers = np.tile(x0, (N, 1))       
    active = np.ones(N, dtype=bool)
    walk_lengths = np.full(N, max_steps, dtype=int)
    u_accum = np.zeros(N)

    for step in range(max_steps):
        if not np.any(active):
            break

        step_seed = None if seed is None else seed * (max_steps + 1) + step

        # --- Fresh RQMC point set: N points in 3D (three uniforms per step) ---
        if qmc_type == "sobol":
            sampler = Sobol(dimension=3, seed=step_seed, randomize=True)
            U = sampler.gen_samples(N)              
        elif qmc_type == "halton":
            sampler = Halton(dimension=3, seed=step_seed, randomize=True)
            U = sampler.gen_samples(N)
        elif qmc_type == "lattice":
            sampler = qmcpy.Lattice(
                dimension=4,
                randomize="SHIFT",
                order="LINEAR",                
                seed=step_seed,
            )
            points = sampler.gen_samples(N)   
            U = points[:, 1:] 
        else:
            raise ValueError(f"Unknown qmc_type: {qmc_type}")

        u1 = U[:, 0]
        u2 = U[:, 1]
        u3 = U[:, 2]

        # --- Sort all walkers by Hilbert key; inactive pushed to end ---
        keys = hilbert_keys_hilbertcurve(walkers)        
        keys = np.where(active, keys, np.inf)
        order = np.argsort(keys, kind="mergesort")

        # Assign by global rank: k-th in sorted order gets (u1[k], u2[k], u3[k])
        assigned_u1 = np.empty(N); assigned_u1[order] = u1
        assigned_u2 = np.empty(N); assigned_u2[order] = u2
        assigned_u3 = np.empty(N); assigned_u3[order] = u3

        # Compute radius to boundary for all walkers
        R_arr = np.array([distance_to_sector_boundary(walkers[i])[0] for i in range(N)])
        
        to_do = active & (R_arr > eps)
        idx = np.where(to_do)[0]
        if idx.size == 0:
            newly_done = active & (R_arr <= eps)
            walk_lengths[newly_done] = step
            active[newly_done] = False
            continue

        # --- Sample inner point y and accumulate Green's term ---
        theta_inner = 2.0 * np.pi * assigned_u2[idx]
        rho = R_arr[idx] * np.sqrt(assigned_u3[idx])

        yx = walkers[idx, 0] + rho * np.cos(theta_inner)
        yy = walkers[idx, 1] + rho * np.sin(theta_inner)

        # Accumulate u_accum for these walkers
        for t, i in enumerate(idx):
            y_i = (float(yx[t]), float(yy[t]))
            x_i = (float(walkers[i, 0]), float(walkers[i, 1]))
            R_i = float(R_arr[i])
            u_accum[i] += (math.pi * R_i**2) * G_disk(x_i, y_i, R_i) * f_rhs(y_i)

        # --- Step walkers to circle surface ---
        theta_outer = 2.0 * np.pi * assigned_u1[idx]
        walkers[idx] += R_arr[idx, None] * np.stack(
            [np.cos(theta_outer), np.sin(theta_outer)], axis=1
        )

        # --- Check boundary and deactivate newly done ---
        R_new = np.array([distance_to_sector_boundary(walkers[i])[0] for i in idx])
        done_local = idx[R_new <= eps]
        walk_lengths[done_local] = step + 1
        active[done_local] = False

    # --- Add boundary term and collect results ---
    results = np.array([
        u_accum[i] + g_boundary((walkers[i, 0], walkers[i, 1]))
        for i in range(N)
    ])

    return np.mean(results), np.mean(walk_lengths)



