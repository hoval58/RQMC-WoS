""" Column-wise Sobol total-indices for gasket Walk-on-Spheres.

This single local script contains four related diagnostics:
  1. Standard non-array MC-WOS.
  2. Standard non-array QMC/RQMC-WOS.
  3. Array-MC with Hilbert sorting.
  4. Array-RQMC with Hilbert sorting.

"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import qmcpy
from qmcpy import Sobol, Halton
from hilbertcurve.hilbertcurve import HilbertCurve


# ===== Constants ========================================
FLT_MAX = np.finfo(np.float32).max
M_PI = np.pi

# LatNet Builder dimension-2 lattice generators.
# Place this JSON file in the same folder as this script.
GENERATOR_FILE_DIM2 = Path(__file__).resolve().parent / "korobov_generators_dim2.json"

_LATTICE_DIM2 = None


def load_lattice_generators_dim2():
    global _LATTICE_DIM2

    if _LATTICE_DIM2 is None:
        try:
            with GENERATOR_FILE_DIM2.open("r") as file:
                _LATTICE_DIM2 = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Could not find 'korobov_generators_dim2.json'. "
                "Place it in the same folder as this script."
            ) from exc

    return _LATTICE_DIM2


def get_lattice_generator_dim2(n):
    """
    Return (a, generating_vector) for the 2D lattice rule

        (1, a) mod n

    using the precomputed LatNet Builder JSON file.
    """
    n = int(n)
    lattice_dim2 = load_lattice_generators_dim2()

    try:
        entry = lattice_dim2[str(n)]
    except KeyError as exc:
        raise KeyError(
            f"No dimension-2 lattice generator stored for N={n} "
            f"in {GENERATOR_FILE_DIM2}."
        ) from exc

    a = int(entry["a"])
    stored_vector = np.asarray(
        entry["generating_vector"],
        dtype=np.int64,
    )

    if n == 4 and a == 1:
        a = 3
        generating_vector = np.array([1, a % n], dtype=np.int64)
    else:
        if not (1 < a < n):
            raise RuntimeError(
                f"Invalid lattice multiplier a={a} for N={n}; "
                "we require 1 < a < N."
            )

        if math.gcd(a, n) != 1:
            raise RuntimeError(
                f"lattice multiplier a={a} is not coprime to N={n}."
            )

        expected_vector = np.array([1, a % n], dtype=np.int64)

        if stored_vector.shape != (2,):
            raise RuntimeError(
                f"Stored generating vector must have length 2; "
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


def lattice_latnet_1d(n, a, generating_vector, seed=None):
    """
    Generate the explicit transition coordinate from the 2D lattice rule

        (1, a) mod n.

    The first coordinate is implicit through the Array-RQMC Hilbert rank.
    """
    n = int(n)
    a = int(a)
    generating_vector = np.asarray(generating_vector, dtype=np.int64)

    if generating_vector.shape != (2,):
        raise ValueError("generating_vector must have shape (2,).")

    expected_vector = np.array([1, a % n], dtype=np.int64)

    if not np.array_equal(generating_vector, expected_vector):
        raise ValueError(
            "The generating vector is inconsistent with a.\n"
            f"Received: {generating_vector.tolist()}\n"
            f"Expected: {expected_vector.tolist()}"
        )

    rng = np.random.default_rng(seed)
    shift = rng.random()

    i = np.arange(n, dtype=np.int64)
    z = int(generating_vector[1])

    return (((i * z) % n) / n + shift) % 1.0


# ===== Scene Data =======================================

# Temperature values (°C)
Touter = 120.0
Tcoolant = 90.0
Toil = 130.0
Toilreturn = 110.0
Tbore = 160.0

# ----- Circles -----
circles = [
    ((0.003010, 0.216652), 0.028835, Tcoolant),
    ((0.003010, 0.134517), 0.009526, Toil),
    ((0.003010, -0.131789), 0.010115, Toil),
    ((0.003010, -0.222832), 0.029699, Tcoolant),
    ((0.003010, -0.336359), 0.059481, Toilreturn),
    ((-0.234978, 0.002839), 0.219154, Tbore),
    ((-0.694982, 0.002839), 0.219154, Tbore),
    ((-0.921277, 0.307325), 0.037897, Toilreturn),
    ((-0.927625, 0.219160), 0.024806, Tcoolant),
    ((-0.829241, 0.246888), 0.024386, Tcoolant),
    ((-0.934773, 0.130514), 0.015977, Tcoolant),
    ((-0.931729, -0.133866), 0.015409, Tcoolant),
    ((-0.922901, -0.216419), 0.023962, Tcoolant),
    ((-0.792235, -0.259206), 0.022056, Tcoolant),
    ((-0.695240, -0.367271), 0.008177, Tcoolant),
    ((-0.555702, -0.233212), 0.019179, Tcoolant),
    ((-0.467730, -0.224006), 0.027652, Tcoolant),
    ((-0.465442, -0.129819), 0.007696, Toil),
    ((-0.465695, 0.134688), 0.009101, Toil),
    ((-0.464799, 0.216560), 0.027652, Tcoolant),
    ((0.240999, 0.002839), 0.219154, Tbore),
    ((0.701002, 0.002839), 0.219154, Tbore),
    ((0.473751, -0.224006), 0.027652, Tcoolant),
    ((0.471463, -0.129819), 0.007696, Toil),
    ((0.471716, 0.134688), 0.009101, Toil),
    ((0.470820, 0.216560), 0.027652, Tcoolant),
    ((-0.097709, -0.252532), 0.022213, Tcoolant),
    ((0.100935, -0.252990), 0.019414, Tcoolant),
    ((0.367999, -0.258228), 0.019100, Tcoolant),
    ((0.519545, -0.275218), 0.027528, Tcoolant),
    ((0.610467, -0.271935), 0.012114, Tcoolant),
    ((0.832751, -0.255299), 0.014503, Tcoolant),
    ((0.934161, -0.224487), 0.018224, Tcoolant),
    ((0.935606, -0.143100), 0.007562, Toil),
    ((0.940940, 0.125145), 0.009584, Toil),
    ((0.937585, 0.214719), 0.021615, Tcoolant),
    ((0.938959, 0.300764), 0.045862, Tcoolant),
    ((0.828571, 0.239434), 0.010003, Tcoolant),
    ((0.561909, 0.242352), 0.018136, Tcoolant),
    ((0.372248, 0.243007), 0.020347, Tcoolant),
    ((-0.720057, 0.367107), 0.009356, Toil),
    ((-0.596684, 0.410074), 0.002858, Toil),
    ((0.505378, 0.406864), 0.006389, Toil),
    ((0.729180, 0.365890), 0.010793, Tcoolant),
    ((-0.175358, -0.407582), 0.004867, Toil),
    ((-0.138091, -0.409801), 0.006806, Toil),
    ((-0.367595, 0.246820), 0.007526, Toil),
    ((0.465414, -0.367129), 0.006136, Toil),
]

# ----- Segments -----
segments = [
    ((0.003010, 0.382168), (-0.540935, 0.382168), Touter),
    ((-0.779611, 0.381423), (-0.973476, 0.381423), Touter),
    ((-1.000000, 0.359642), (-1.000000, -0.225414), Touter),
    ((-0.864032, -0.318778), (-0.757349, -0.318778), Touter),
    ((-0.644130, -0.323580), (-0.558650, -0.323580), Touter),
    ((-0.512375, -0.395956), (-0.353712, -0.395956), Touter),
    ((-0.311812, -0.433059), (0.013181, -0.433059), Touter),
    ((0.127324, -0.326990), (0.412948, -0.326990), Touter),
    ((0.516828, -0.331144), (0.841977, -0.331144), Touter),
    ((1.000000, -0.252687), (1.000000, 0.348212), Touter),
    ((0.973883, 0.377747), (0.784100, 0.377747), Touter),
    ((0.672766, 0.377580), (0.559237, 0.377580), Touter),
    ((0.003010, 0.382168), (0.458203, 0.382168), Touter),
]

# ----- Arcs -----
arcs = [
    ((-0.542620, 0.410817), 0.028699, 3.219131, 4.771143, Touter),
    ((-0.596678, 0.403812), 0.025891, 0.185763, 3.052984, Touter),
    ((-0.654496, 0.415072), 0.033260, 3.671326, 6.010146, Touter),
    ((-0.718273, 0.352936), 0.057316, 0.912246, 2.362373, Touter),
    ((-0.781413, 0.408387), 0.027024, 4.779128, 5.686995, Touter),
    ((-0.975190, 0.356470), 0.025012, 1.502182, 3.014421, Touter),
    ((-0.846678, -0.147817), 0.171840, 3.610110, 4.611229, Touter),
    ((-0.752664, -0.351666), 0.033220, 0.276949, 1.712292, Touter),
    ((-0.552554, -0.344665), 0.168169, 3.129210, 3.330616, Touter),
    ((-0.694308, -0.365316), 0.025853, 3.578877, 6.044036, Touter),
    ((0.895765, -0.457167), 1.567301, 3.067765, 3.086868, Touter),
    ((-0.643320, -0.348498), 0.024932, 1.603287, 2.859635, Touter),
    ((-0.565301, -0.349629), 0.026885, 0.048340, 1.320837, Touter),
    ((-0.450123, -0.358840), 0.088947, 3.023160, 3.385666, Touter),
    ((-0.504409, -0.357350), 0.039419, 3.764095, 4.508911, Touter),
    ((-0.353769, -0.424758), 0.028802, 0.331141, 1.568823, Touter),
    ((-0.300675, -0.408814), 0.026681, 3.390764, 4.281779, Touter),
    ((0.003242, -0.334984), 0.098577, 4.813382, 6.117513, Touter),
    ((0.131069, -0.358132), 0.031365, 1.690489, 2.920101, Touter),
    ((0.412547, -0.356629), 0.029641, 6.226018, 7.840459, Touter),
    ((0.468420, -0.365298), 0.027191, 2.882139, 6.663804, Touter),
    ((0.517652, -0.355117), 0.023987, 1.605162, 3.144953, Touter),
    ((0.845299, -0.139467), 0.191706, 4.695062, 5.651394, Touter),
    ((0.964581, 0.343207), 0.035771, 0.140369, 1.307722, Touter),
    ((0.804737, 0.437245), 0.062976, 3.800252, 4.378524, Touter),
    ((0.727896, 0.356738), 0.049920, 0.998398, 2.526422, Touter),
    ((0.670863, 0.397937), 0.020446, 4.805591, 5.632154, Touter),
    ((0.557371, 0.407608), 0.030086, 3.365988, 4.774440, Touter),
    ((0.500652, 0.405320), 0.027739, 6.123652, 9.641687, Touter),
    ((0.451023, 0.404043), 0.023023, 5.029538, 6.077924, Touter),
    ((-0.188931, 0.081900), 0.192090, 0.954758, 1.355593, Toilreturn),
    ((-0.194886, 0.055454), 0.192090, 1.003822, 1.355593, Toilreturn),
    ((-0.145988, 0.255232), 0.014454, 1.704204, 4.136017, Toilreturn),
    ((-0.085750, 0.228681), 0.012685, 4.222547, 7.190830, Toilreturn),
    ((-0.648934, 0.081900), 0.192090, 0.954758, 1.355593, Toilreturn),
    ((-0.654889, 0.055454), 0.192090, 1.003822, 1.355593, Toilreturn),
    ((-0.605992, 0.255232), 0.014454, 1.704204, 4.136017, Toilreturn),
    ((-0.545753, 0.228681), 0.012685, 4.222547, 7.190830, Toilreturn),
    ((-0.376205, -0.448074), 0.198345, 1.788252, 2.217562, Toilreturn),
    ((-0.563286, 0.006008), 0.388384, 5.018011, 5.265426, Toilreturn),
    ((-0.383743, -0.302301), 0.033022, 5.546757, 7.050405, Toilreturn),
    ((-2.316609, -3.558303), 3.818346, 1.032784, 1.045697, Toilreturn),
    ((-0.410715, -0.281352), 0.028196, 1.274307, 1.868952, Toilreturn),
    ((-0.458346, -0.318667), 0.047240, 2.483835, 4.967462, Toilreturn),
]


# ====== Geometric queries ================================

def distanceCircle_vec(xs, C):
    """xs: (n,2)"""
    xs = np.atleast_2d(xs)
    center, radius, _ = C
    return np.abs(np.linalg.norm(xs - np.array(center), axis=1) - radius)

def distanceSegment_vec(xs, S):
    """xs: (n,2)"""
    xs = np.atleast_2d(xs)
    start, end, _ = S
    pa = xs - np.array(start)
    ba = np.array(end) - np.array(start)
    h = np.clip((pa @ ba) / np.dot(ba, ba), 0.0, 1.0)
    return np.linalg.norm(pa - np.outer(h, ba), axis=1)

def distanceArc_vec(xs, A):
    """xs: (n,2)"""
    xs = np.atleast_2d(xs)
    center, radius, startAngle, endAngle, _ = A
    phi = (endAngle - startAngle) / 2
    psi = (endAngle + startAngle) / 2
    s = np.cos(psi)
    c = np.sin(psi)
    p0 = xs - np.array(center)
    qx = c * p0[:, 0] - s * p0[:, 1]
    qy = s * p0[:, 0] + c * p0[:, 1]
    qx = np.abs(qx)
    sc = np.array([np.sin(phi), np.cos(phi)])
    cond = sc[1] * qx > sc[0] * qy
    dist = np.where(
        cond,
        np.linalg.norm(np.stack([qx, qy], axis=1) - sc * radius, axis=1),
        np.abs(np.sqrt(qx**2 + qy**2) - radius),
    )
    return dist

def distanceBoundary_vec(xs):
    """xs: (n,2), returns (n,) array of distances"""
    xs = np.atleast_2d(xs)
    d = np.full(len(xs), FLT_MAX)
    for C in circles:
        d = np.minimum(d, distanceCircle_vec(xs, C))
    for S in segments:
        d = np.minimum(d, distanceSegment_vec(xs, S))
    for A in arcs:
        d = np.minimum(d, distanceArc_vec(xs, A))
    return d

def boundaryValue(x):
    dmin = FLT_MAX
    g = 0.0
    for C in circles:
        di = distanceCircle_vec(x, C).item()   
        if di < dmin:
            dmin = di
            g = C[2]
    for S in segments:
        di = distanceSegment_vec(x, S).item()
        if di < dmin:
            dmin = di
            g = S[2]
    for A in arcs:
        di = distanceArc_vec(x, A).item()
        if di < dmin:
            dmin = di
            g = A[4]
    return g


# ====== Hilbert curve sorting ============================

p = 13
hc = HilbertCurve(p, 2)
nside = 1 << p

def disk_to_unit_square_xy(xy: np.ndarray) -> np.ndarray:
    uv = 0.5 * (xy + 1.0)
    eps = np.nextafter(1.0, 0.0)
    return np.clip(uv, 0.0, eps)

def hilbert_keys_hilbertcurve(xy: np.ndarray) -> np.ndarray:
    uv = disk_to_unit_square_xy(xy)
    ij = np.floor(uv * nside).astype(int)
    ij = np.clip(ij, 0, nside - 1)
    return np.array([hc.distance_from_point([int(ix), int(iy)]) for ix, iy in ij])


# ====== Array-RQMC step vector ============================

def _array_rqmc_step_vector(
    N,
    qmc_type="sobol",
    seed=None,
    lattice_a=None,
    lattice_vector=None,
):
    """Generate one randomized-QMC transition vector in [0, 1)^N."""
    if qmc_type == "sobol":
        sampler = Sobol(dimension=1, seed=seed, randomize=True)
        u = sampler.gen_samples(N).reshape(-1)

    elif qmc_type == "halton":
        sampler = Halton(dimension=1, seed=seed, randomize=True)
        u = sampler.gen_samples(N).reshape(-1)

    elif qmc_type == "lattice":
        if lattice_a is None or lattice_vector is None:
            raise ValueError(
                "lattice_a and lattice_vector are required "
                "when qmc_type='lattice'."
            )

        u = lattice_latnet_1d(
            n=N,
            a=lattice_a,
            generating_vector=lattice_vector,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown qmc_type: {qmc_type}. "
            "Choose from 'sobol', 'halton', or 'lattice'."
        )

    return u


# ====== Array-RQMC matrix ================================

def generate_array_rqmc_matrix(N, max_steps, qmc_type="sobol", seed=0):
    X = np.empty((max_steps, N), dtype=float)

    lattice_a = None
    lattice_vector = None

    if qmc_type == "lattice":
        lattice_a, lattice_vector = get_lattice_generator_dim2(N)

    for step in range(max_steps):
        step_seed = None if seed is None else seed + step

        X[step, :] = _array_rqmc_step_vector(
            N,
            qmc_type=qmc_type,
            seed=step_seed,
            lattice_a=lattice_a,
            lattice_vector=lattice_vector,
        )

    return X


# ====== WoS core routines ================================

def _advance_one_step_active_only(walkers, active, u_row, eps):
    active_idx = np.flatnonzero(active)
    n_active = active_idx.size
    if n_active == 0:
        return walkers, active

    walkers_active = walkers[active_idx].copy()

    keys = hilbert_keys_hilbertcurve(walkers_active)
    perm = np.argsort(keys, kind="mergesort")

    assigned_u_active = np.empty(n_active, dtype=float)
    assigned_u_active[perm] = u_row[:n_active]

    r = distanceBoundary_vec(walkers_active)
    to_move = r > eps

    theta = 2.0 * np.pi * assigned_u_active
    walkers_active[to_move] += r[to_move, None] * np.stack(
        [np.cos(theta[to_move]), np.sin(theta[to_move])], axis=1
    )

    r_new = distanceBoundary_vec(walkers_active)
    done_local = r_new <= eps

    walkers[active_idx] = walkers_active
    active[active_idx[done_local]] = False
    return walkers, active


def _mean_boundary_value(walkers):
    return np.mean([boundaryValue(w) for w in walkers])


def _simulate_A_and_prefix_states(x0, A, eps, analysis_steps):
    """
    Run WoS with driver matrix A (shape max_steps x N).
    Save walker states before steps 0, ..., analysis_steps-1.
    """
    x0 = np.array(x0, dtype=float)
    max_steps, N = A.shape

    walkers = np.tile(x0, (N, 1))
    active = np.ones(N, dtype=bool)

    prefix_states = []

    for step in range(max_steps):
        if step < analysis_steps:
            prefix_states.append((walkers.copy(), active.copy()))

        if not np.any(active):
            break

        walkers, active = _advance_one_step_active_only(
            walkers, active, A[step], eps
        )

    YA = _mean_boundary_value(walkers)
    return YA, prefix_states


def _continue_from_prefix_state(walkers0, active0, A, k, replacement_row, eps):
    """
    Resume from state before step k, apply replacement_row at step k,
    then continue with A[k+1], ..., A[max_steps-1].
    """
    walkers = walkers0.copy()
    active = active0.copy()
    max_steps, _ = A.shape

    if not np.any(active):
        return _mean_boundary_value(walkers)

    walkers, active = _advance_one_step_active_only(
        walkers, active, replacement_row, eps
    )

    for step in range(k + 1, max_steps):
        if not np.any(active):
            break
        walkers, active = _advance_one_step_active_only(
            walkers, active, A[step], eps
        )

    return _mean_boundary_value(walkers)


# ====== Sobol total indices ==============================

# Array-RQMC 

def sobol_total_indices_array_rqmc_steps(
    x0,
    N,
    analysis_steps,
    max_steps,
    M,
    qmc_type="sobol",
    eps=1e-3,
    seed=1234,
):
    """
    Total Sobol indices for the first `analysis_steps` RQMC step-vectors,
    while simulating WoS up to `max_steps`.
    """
    if analysis_steps > max_steps:
        raise ValueError("analysis_steps must be <= max_steps")

    YA  = np.empty(M, dtype=float)
    YAB = np.empty((M, analysis_steps), dtype=float)

    base_seed = 0 if seed is None else seed

    # Non-overlapping seed blocks of size 2*max_steps per replicate
    for m in range(M):
        A_seed = base_seed + (2 * m) * max_steps
        B_seed = base_seed + (2 * m + 1) * max_steps

        A = generate_array_rqmc_matrix(
            N, max_steps, qmc_type=qmc_type, seed=A_seed,
        )
        B = generate_array_rqmc_matrix(
            N, max_steps, qmc_type=qmc_type, seed=B_seed,
        )

        YA[m], prefix_states = _simulate_A_and_prefix_states(
            x0, A, eps, analysis_steps
        )

        for k in range(analysis_steps):
            walkers_k, active_k = prefix_states[k]

            if not np.any(active_k):
                YAB[m, k:] = YA[m]
                break

            YAB[m, k] = _continue_from_prefix_state(
                walkers_k, active_k, A, k, B[k], eps
            )

    variance     = np.var(YA, ddof=1)
    tau2_total   = 0.5 * np.mean((YA[:, None] - YAB) ** 2, axis=0)
    total_indices = tau2_total / variance

    return {
        "variance":                     variance,
        "tau2_total":                   tau2_total,
        "total_indices":                total_indices,
        "mean_dimension_estimate_partial": np.sum(total_indices),
        "YA":                           YA,
        "YAB":                          YAB,
        "analysis_steps":               analysis_steps,
        "max_steps":                    max_steps,
    }



# ====== Sobol total indices: Array-MC ====================

def generate_array_mc_matrix(N, max_steps, rng):
    """Plain iid U[0, 1] matrix, shape (max_steps, N)."""
    return rng.random((max_steps, N))


def sobol_total_indices_array_mc_steps(
    x0,
    N,
    analysis_steps,
    max_steps,
    M,
    eps=1e-3,
    seed=1234,
):
    """
    Total Sobol indices for the first `analysis_steps` step-vectors,
    using Array-MC: iid uniforms plus the same Hilbert sorting rule.
    """
    if analysis_steps > max_steps:
        raise ValueError("analysis_steps must be <= max_steps")

    YA = np.empty(M, dtype=float)
    YAB = np.empty((M, analysis_steps), dtype=float)

    rng = np.random.default_rng(seed)

    for m in range(M):
        A = generate_array_mc_matrix(N, max_steps, rng)
        B = generate_array_mc_matrix(N, max_steps, rng)

        YA[m], prefix_states = _simulate_A_and_prefix_states(
            x0, A, eps, analysis_steps
        )

        for k in range(analysis_steps):
            walkers_k, active_k = prefix_states[k]

            if not np.any(active_k):
                YAB[m, k:] = YA[m]
                break

            YAB[m, k] = _continue_from_prefix_state(
                walkers_k, active_k, A, k, B[k], eps
            )

    variance = np.var(YA, ddof=1)
    tau2_total = 0.5 * np.mean((YA[:, None] - YAB) ** 2, axis=0)
    total_indices = tau2_total / variance

    return {
        "variance": variance,
        "tau2_total": tau2_total,
        "total_indices": total_indices,
        "mean_dimension_estimate_partial": np.sum(total_indices),
        "YA": YA,
        "YAB": YAB,
        "analysis_steps": analysis_steps,
        "max_steps": max_steps,
    }



# ====== Sobol total indices: standard non-array MC ========

def generate_mc_matrix(N, max_steps, seed=0):
    """
    Generate an iid Monte Carlo driver matrix of shape (max_steps, N).

    Row k gives the step-k uniforms for all walkers. No Hilbert sorting.
    """
    rng = np.random.default_rng(seed)
    return rng.random((max_steps, N))


def sobol_total_indices_mc_steps(
    x0,
    N,
    analysis_steps,
    max_steps,
    M,
    eps=1e-3,
    seed=1234,
):
    """
    Total Sobol indices for standard plain-MC WoS.

    Group k is the vector of N iid uniforms used at step k across walkers.
    No Hilbert sorting is used.
    """
    if analysis_steps > max_steps:
        raise ValueError("analysis_steps must be <= max_steps")
    if M < 2:
        raise ValueError("M must be at least 2 to estimate variance")

    YA = np.empty(M, dtype=float)
    YAB = np.empty((M, analysis_steps), dtype=float)

    base_seed = 0 if seed is None else seed

    for m in range(M):
        A_seed = base_seed + 2 * m
        B_seed = base_seed + 2 * m + 1

        A = generate_mc_matrix(N, max_steps, seed=A_seed)
        B = generate_mc_matrix(N, max_steps, seed=B_seed)

        YA[m], prefix_states = _simulate_A_and_prefix_states_qmc(
            x0, A, eps, analysis_steps
        )

        for k in range(analysis_steps):
            walkers_k, active_k = prefix_states[k]

            if not np.any(active_k):
                YAB[m, k:] = YA[m]
                break

            YAB[m, k] = _continue_from_prefix_state_qmc(
                walkers_k, active_k, A, k, B[k], eps
            )

    variance = np.var(YA, ddof=1)
    if variance <= 0:
        raise ValueError("Estimated variance is non-positive; Sobol indices are undefined")

    tau2_total = 0.5 * np.mean((YA[:, None] - YAB) ** 2, axis=0)
    total_indices = tau2_total / variance

    return {
        "variance": variance,
        "tau2_total": tau2_total,
        "total_indices": total_indices,
        "mean_dimension_estimate_partial": np.sum(total_indices),
        "YA": YA,
        "YAB": YAB,
        "analysis_steps": analysis_steps,
        "max_steps": max_steps,
    }


# ====== Sobol total indices: standard non-array RQMC ======

def generate_qmc_matrix(N, max_steps, qmc_type="sobol", seed=0):
    """
    Generate a non-array QMC driver matrix of shape (max_steps, N).

    Row k gives the step-k uniforms for all walkers. No Hilbert sorting
    is used in the standard QMC method.
    """
    if qmc_type == "sobol":
        sampler = Sobol(dimension=max_steps, seed=seed, randomize=True)
        points = sampler.gen_samples(N)

    elif qmc_type == "halton":
        sampler = Halton(dimension=max_steps, seed=seed, randomize=True)
        points = sampler.gen_samples(N)

    elif qmc_type == "lattice":
        sampler = qmcpy.Lattice(
            dimension=max_steps,
            randomize="SHIFT",
            seed=seed,
        )
        points = sampler.gen_samples(n=N)

    else:
        raise ValueError(
            f"Unknown qmc_type: {qmc_type}. "
            "Choose from 'sobol', 'halton', or 'lattice'."
        )

    return points.T


def _advance_one_step_qmc(walkers, active, u_row, eps):
    """
    Advance active walkers by one WoS step without Hilbert sorting.
    Walker i uses u_row[i].
    """
    active_idx = np.flatnonzero(active)
    if active_idx.size == 0:
        return walkers, active

    walkers_active = walkers[active_idx].copy()
    u_active = u_row[active_idx]

    r = distanceBoundary_vec(walkers_active)
    to_move = r > eps

    theta = 2.0 * np.pi * u_active
    walkers_active[to_move] += r[to_move, None] * np.stack(
        [np.cos(theta[to_move]), np.sin(theta[to_move])], axis=1
    )

    r_new = distanceBoundary_vec(walkers_active)
    done_local = r_new <= eps

    walkers[active_idx] = walkers_active
    active[active_idx[done_local]] = False
    return walkers, active


def _simulate_A_and_prefix_states_qmc(x0, A, eps, analysis_steps):
    """Run standard QMC WoS and save prefix states before selected steps."""
    x0 = np.array(x0, dtype=float)
    max_steps, N = A.shape

    walkers = np.tile(x0, (N, 1))
    active = np.ones(N, dtype=bool)
    prefix_states = []

    for step in range(max_steps):
        if step < analysis_steps:
            prefix_states.append((walkers.copy(), active.copy()))

        if not np.any(active):
            break

        walkers, active = _advance_one_step_qmc(
            walkers, active, A[step], eps
        )

    YA = _mean_boundary_value(walkers)
    return YA, prefix_states


def _continue_from_prefix_state_qmc(walkers0, active0, A, k, replacement_row, eps):
    """
    Resume standard RQMC from the state before step k, apply replacement_row,
    then continue with A[k+1], ..., A[max_steps-1].
    """
    walkers = walkers0.copy()
    active = active0.copy()
    max_steps, _ = A.shape

    if not np.any(active):
        return _mean_boundary_value(walkers)

    walkers, active = _advance_one_step_qmc(
        walkers, active, replacement_row, eps
    )

    for step in range(k + 1, max_steps):
        if not np.any(active):
            break
        walkers, active = _advance_one_step_qmc(
            walkers, active, A[step], eps
        )

    return _mean_boundary_value(walkers)


def sobol_total_indices_rqmc_steps(
    x0,
    N,
    analysis_steps,
    max_steps,
    M,
    qmc_type="sobol",
    eps=1e-3,
    seed=1234,
):
    """
    Total Sobol indices for standard non-array RQMC WoS.

    Group k is the vector of N uniforms used at step k across walkers.
    """
    if analysis_steps > max_steps:
        raise ValueError("analysis_steps must be <= max_steps")

    YA = np.empty(M, dtype=float)
    YAB = np.empty((M, analysis_steps), dtype=float)

    base_seed = 0 if seed is None else seed

    for m in range(M):
        A_seed = base_seed + 2 * m
        B_seed = base_seed + 2 * m + 1

        A = generate_qmc_matrix(N, max_steps, qmc_type=qmc_type, seed=A_seed)
        B = generate_qmc_matrix(N, max_steps, qmc_type=qmc_type, seed=B_seed)

        YA[m], prefix_states = _simulate_A_and_prefix_states_qmc(
            x0, A, eps, analysis_steps
        )

        for k in range(analysis_steps):
            walkers_k, active_k = prefix_states[k]

            if not np.any(active_k):
                YAB[m, k:] = YA[m]
                break

            YAB[m, k] = _continue_from_prefix_state_qmc(
                walkers_k, active_k, A, k, B[k], eps
            )

    variance = np.var(YA, ddof=1)
    tau2_total = 0.5 * np.mean((YA[:, None] - YAB) ** 2, axis=0)
    total_indices = tau2_total / variance

    return {
        "variance": variance,
        "tau2_total": tau2_total,
        "total_indices": total_indices,
        "mean_dimension_estimate_partial": np.sum(total_indices),
        "YA": YA,
        "YAB": YAB,
        "analysis_steps": analysis_steps,
        "max_steps": max_steps,
    }


# ====== Entry point ======================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate total Sobol indices for gasket Walk-on-Spheres "
            "using MC, RQMC, Array-MC, or Array-RQMC."
        )
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["mc", "rqmc", "array_mc", "array_rqmc"],
        default="array_rqmc",
        help=(
            "mc: standard iid Monte Carlo; "
            "rqmc: standard non-array RQMC; "
            "array_mc: Hilbert sorting with iid uniforms; "
            "array_rqmc: Hilbert sorting with RQMC. "
        ),
    )
    parser.add_argument(
        "--qmc_type",
        type=str,
        choices=["sobol", "halton", "lattice"],
        default="lattice",
        help=(
            "For method=array_rqmc, lattice uses korobov_generators_dim2.json. "
            "For method=rqmc, lattice uses the QMCPy lattice. "
            "Ignored for method=mc and method=array_mc."
        ),
    )
    parser.add_argument("--M", type=int, default=800)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--analysis_steps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--outdir",
        type=str,
        default="results_gasket_sobol_indices",
    )

    args = parser.parse_args()

    if args.analysis_steps > args.max_steps:
        raise ValueError("analysis_steps must be <= max_steps.")

    x0 = [0.240999, 0.3]
    method = args.method

    print(
        f"Running method={method}, qmc_type={args.qmc_type}, "
        f"N={args.N}, M={args.M}",
        flush=True,
    )

    if method == "mc":
        out = sobol_total_indices_mc_steps(
            x0=x0,
            N=args.N,
            analysis_steps=args.analysis_steps,
            max_steps=args.max_steps,
            M=args.M,
            eps=args.eps,
            seed=args.seed,
        )

    elif method == "array_rqmc":
        out = sobol_total_indices_array_rqmc_steps(
            x0=x0,
            N=args.N,
            analysis_steps=args.analysis_steps,
            max_steps=args.max_steps,
            M=args.M,
            qmc_type=args.qmc_type,
            eps=args.eps,
            seed=args.seed,
        )

    elif method == "array_mc":
        out = sobol_total_indices_array_mc_steps(
            x0=x0,
            N=args.N,
            analysis_steps=args.analysis_steps,
            max_steps=args.max_steps,
            M=args.M,
            eps=args.eps,
            seed=args.seed,
        )

    elif method == "rqmc":
        out = sobol_total_indices_rqmc_steps(
            x0=x0,
            N=args.N,
            analysis_steps=args.analysis_steps,
            max_steps=args.max_steps,
            M=args.M,
            qmc_type=args.qmc_type,
            eps=args.eps,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    results = {
        "variance": out["variance"],
        "tau2_total": out["tau2_total"],
        "total_indices": out["total_indices"],
        "mean_dimension_estimate_partial": (
            out["mean_dimension_estimate_partial"]
        ),
        "YA": out["YA"],
        "YAB": out["YAB"],
        "analysis_steps": out["analysis_steps"],
        "max_steps": out["max_steps"],
        "M": args.M,
        "N": args.N,
        "method": method,
        "qmc_type": None if method in ["mc", "array_mc"] else args.qmc_type,
        "eps": args.eps,
        "seed": args.seed,
        "x0": x0,
    }

    if method == "array_rqmc" and args.qmc_type == "lattice":
        lattice_a, lattice_vector = get_lattice_generator_dim2(args.N)
        results["lattice_a"] = lattice_a
        results["lattice_generating_vector"] = lattice_vector.tolist()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if method in ["mc", "array_mc"]:
        method_label = method
    else:
        method_label = f"{method}_{args.qmc_type}"

    filename = (
        f"sobol_indices_{method_label}"
        f"_N={args.N}_M={args.M}"
        f"_K={args.analysis_steps}.pickle"
    )
    outpath = outdir / filename

    with outpath.open("wb") as file:
        pickle.dump(results, file)

    print(f"Saved results to {outpath}", flush=True)
    print(
        "Partial mean-dimension estimate: "
        f"{results['mean_dimension_estimate_partial']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
