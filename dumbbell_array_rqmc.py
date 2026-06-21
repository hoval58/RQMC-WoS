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
    """Return the precomputed dimension-2 Korobov multiplier for n."""
    n = int(n)
    try:
        a = int(KOROBOV_DIM2[str(n)]["a"])
    except KeyError as exc:
        raise KeyError(
            f"No dimension-2 Korobov generator stored for N={n} "
            f"in {GENERATOR_FILE_DIM2}."
        ) from exc

    # For N=4, a=1 and a=3 may tie under P2. Use a>1 as in the paper.
    if n == 4 and a == 1:
        a = 3

    if not (1 < a < n) or math.gcd(a, n) != 1:
        raise RuntimeError(
            f"Invalid stored Korobov generator a={a} for N={n}."
        )

    return a


#### HELPER FUNCTIONS

def disk_to_unit_square_xy(xy: np.ndarray) -> np.ndarray:
    """
    xy: (n,2) array with points in unit disk (x^2+y^2 <= 1)
    returns uv in [0,1)^2
    """
    uv = 0.5 * (xy + 1.0)             # maps [-1,1] -> [0,1]
    eps = np.nextafter(1.0, 0.0)      # largest float < 1
    return np.clip(uv, 0.0, eps)

p = 13
hc = HilbertCurve(p, 2)
nside = 1 << p

def hilbert_keys_hilbertcurve(xy: np.ndarray) -> np.ndarray:
    uv = disk_to_unit_square_xy(xy)
    ij = np.floor(uv * nside).astype(int)
    ij = np.clip(ij, 0, nside - 1)
    return np.array([hc.distance_from_point([int(ix), int(iy)]) for ix, iy in ij])


def _closest_point_on_circle(px, py, cx, cy, R):
    vx, vy = px - cx, py - cy
    d = np.hypot(vx, vy)
    if d == 0.0:
        return (cx + R, cy), R
    qx = cx + (vx / d) * R
    qy = cy + (vy / d) * R
    return (qx, qy), abs(d - R)

def _point_in_circle_strict(x, y, cx, cy, R):
    # strict interior test; points exactly on circle are considered not interior here
    return (x - cx)**2 + (y - cy)**2 < R**2

def distance_to_boundary_dumbbell(px_py, L, R, w):
    """
    Distance from point (px,py) to boundary of union:
      rectangle [-L,L] x [-w/2,w/2]  U  disks centered at (-L,0) and (L,0) radius R.

    Correctly handles exposed vs hidden parts of boundaries.
    """
    px, py = px_py
    half_w = w / 2.0

    # --- helper: circle coverage of top/bottom edge ---
    # For fixed y = +half_w (top), points x satisfying (x - cx)^2 + (half_w)^2 <= R^2 are covered by that circle.
    s = 0.0
    if R > half_w:
        s = np.sqrt(max(0.0, R**2 - half_w**2))
    # left circle covers interval on top: [-L - s, -L + s]
    left_cover = (-L - s, -L + s)
    # right circle covers interval on top: [ L - s,  L + s]
    right_cover = ( L - s,  L + s)

    # Intersection with rectangle x-range [-L, L] gives covered subintervals
    def intersect_interval(a,b, lo=-L, hi=L):
        return (max(a, lo), min(b, hi))

    cov1 = intersect_interval(left_cover[0], left_cover[1])
    cov2 = intersect_interval(right_cover[0], right_cover[1])

    # build union of covered intervals on top edge within [-L,L]
    covered = []
    for a,b in (cov1, cov2):
        if a <= b:
            covered.append((a,b))
    covered = sorted(covered)

    # merge if overlapping
    merged = []
    for seg in covered:
        if not merged:
            merged.append(seg)
        else:
            a,b = seg
            la,lb = merged[-1]
            if a <= lb + 1e-14:
                merged[-1] = (la, max(lb, b))
            else:
                merged.append(seg)

    covered_intervals = merged  # list of (a,b) covered on top edge

    # function to find closest x' on top edge that is not covered (within [-L,L])
    def nearest_exposed_on_edge(px, edge_y):
        # allowed set is [-L,L] minus covered_intervals (same for top and bottom because squared term same)
        # if there are no covered intervals, nearest is clamp(px,-L,L)
        candidates_x = []

        # if no coverage -> entire edge exposed
        if not covered_intervals:
            x_clamped = np.clip(px, -L, L)
            return x_clamped, np.hypot(px - x_clamped, py - edge_y)

        # build list of exposed intervals (complement inside [-L,L])
        exposed = []
        cur = -L
        for (a,b) in covered_intervals:
            if cur < a - 1e-14:
                exposed.append((cur, a))
            cur = max(cur, b)
        if cur < L - 1e-14:
            exposed.append((cur, L))

        # If there are exposed intervals, find nearest point in them to px
        if exposed:
            # project px into each interval (clamp) and pick nearest
            best_x = None
            best_d2 = None
            for a,b in exposed:
                xproj = np.clip(px, a, b)
                d2 = (px - xproj)**2 + (py - edge_y)**2
                if (best_d2 is None) or (d2 < best_d2):
                    best_d2 = d2
                    best_x = xproj
            return best_x, np.sqrt(best_d2)

        # If no exposed portion on the segment (fully covered), return None
        return None, np.inf

    # --- Candidate 1 & 2: nearest exposed points on left/right circle arcs ---
    candidates = []

    # left circle
    (qlx, qly), dl = _closest_point_on_circle(px, py, -L, 0.0, R)
    # this circle-boundary point is exposed if it is not strictly inside rectangle interior
    # i.e., exposed if |qly| >= half_w or |qlx| > L (on outside arc),
    # or if it's on rectangle boundary exactly (we consider that exposed)
    if (abs(qly) >= half_w - 1e-14) or (abs(qlx) > L + 1e-14) or (abs(qly) <= half_w + 1e-14 and abs(qlx) >= L - 1e-14):
        # double-check: ensure it's not strictly inside other circle (right) either
        if not _point_in_circle_strict(qlx, qly, L, 0.0, R):
            candidates.append(dl)

    # right circle
    (qrx, qry), dr = _closest_point_on_circle(px, py, L, 0.0, R)
    if (abs(qry) >= half_w - 1e-14) or (abs(qrx) > L + 1e-14) or (abs(qry) <= half_w + 1e-14 and abs(qrx) >= L - 1e-14):
        if not _point_in_circle_strict(qrx, qry, -L, 0.0, R):
            candidates.append(dr)

    # --- Candidate 3 & 4: nearest exposed on top/bottom rectangle edges ---
    # top y = +half_w
    xtop, dtop = nearest_exposed_on_edge(px, +half_w)
    if dtop < np.inf:
        candidates.append(dtop)
    # bottom y = -half_w
    xbot, dbot = nearest_exposed_on_edge(px, -half_w)
    if dbot < np.inf:
        candidates.append(dbot)

    # --- If we have candidates, return the minimum ---
    if candidates:
        return float(min(candidates))

    # --- If no exposed candidates found ---
    # compute distance to complement explicitly:
    # if outside union, distance to enter union:
    inside_rect = (-L <= px <= L) and (abs(py) <= half_w)
    inside_left = (px + L)**2 + py**2 <= R**2
    inside_right = (px - L)**2 + py**2 <= R**2
    inside_union = inside_rect or inside_left or inside_right

    if not inside_union:
        dx = max(0.0, abs(px) - L)
        dy = max(0.0, abs(py) - half_w)
        dist_to_rect = np.hypot(dx, dy)
        dist_to_left = max(0.0, np.hypot(px + L, py) - R)
        dist_to_right = max(0.0, np.hypot(px - L, py) - R)
        return float(min(dist_to_rect, dist_to_left, dist_to_right))
    else:
        # inside but every local boundary candidate is hidden (rare) — return min distance to component boundaries
        return float(min(abs(np.hypot(px + L, py) - R),
                         abs(np.hypot(px - L, py) - R),
                         half_w - abs(py)))


def wos_dumbbell_mc(x0, w,L,R,N=1000, max_steps=500, eps=1e-4, seed=None):
    """
    WOS estimator with dumbbell domain

    Parameters
    x0: point where want to estimate solution to PDE
    w: width of the rectangular region of the dumbbell
    L: half length of the rectangular region
    R: radius of the circular lobes of the dumbbell
    N: number of trajectories
    max_steps: max steps allowed for each walk
    eps: termination threshold
    seed: for reproducibility

    """
    rng = np.random.default_rng(seed)
    results = []

    for i in range(N):
        x = np.array(x0, dtype=float)
        acc = 0.0

        for step in range(max_steps):
            r = distance_to_boundary_dumbbell(x, L, R, w)

            if r <= eps:
                break

            # expected contribution from current ball (coming from source term)
            acc += (r**2) / 2.0

            # pick random direction
            theta = rng.uniform(0.0, 2.0 * np.pi)
            x = x + r * np.array([np.cos(theta), np.sin(theta)])

        results.append(acc)

    return np.mean(results)

def wos_dumbbell_rqmc(x0, w, L, R, N, type, max_steps=1000, eps=1e-4, seed=None):
    """
    Walk-on-Spheres in the unit disk using QMC (Sobol) for N trajectories.

    Parameters:
        x0: point where want to estimate solution to PDE
        w: width of the rectangular region of the dumbbell
        L: half length of the rectangular region
        R: radius of the circular lobes of the dumbbell
        N: number of trajectories
        max_steps: max steps allowed for each walk
        eps: termination threshold
        seed: optional Sobol scrambling seed
    Returns:
        List of boundary values for each trajectory
    """
    x0 = np.array(x0, dtype=float)
    results = []
    if type=="sobol":
      sobol = Sobol(dimension=max_steps, seed=seed, randomize=True)
      points = sobol.gen_samples(N)  
    if type=="halton":
      halton=Halton(dimension=max_steps, seed=seed, randomize=True)
      points=halton.gen_samples(N)
    if type=="niederreiter":
      nied = qmcpy.DigitalNet(dimension=max_steps, generating_matrices="ebert_osisiogu.niederreiter_32.txt",randomize="LMS_DS",seed=seed)
      points = nied.gen_samples(n_min=0, n_max=N)
    if type=="lattice":
      lattice = qmcpy.Lattice(dimension=max_steps, randomize='SHIFT',seed=seed)
      points = lattice.gen_samples(n=N)


    for i in range(N):
        x = x0.copy()
        acc=0
        for u in points[i]:
            r=distance_to_boundary_dumbbell(x, L, R, w)
            if r <= eps:
                break
            acc+=(r**2)/2
            theta = 2 * np.pi * u
            x += r * np.array([np.cos(theta), np.sin(theta)])

        results.append(acc)

    return (np.mean(results))


def domain_to_unit_square_xy(xy, L, R, w):
    xy = np.asarray(xy, dtype=float)
    ymax = max(R, w / 2.0)

    xmin, xmax = -L - R, L + R
    ymin, ymax = -ymax, ymax

    uv = np.empty_like(xy)
    uv[:, 0] = (xy[:, 0] - xmin) / (xmax - xmin)
    uv[:, 1] = (xy[:, 1] - ymin) / (ymax - ymin)

    eps1 = np.nextafter(1.0, 0.0)
    return np.clip(uv, 0.0, eps1)


def hilbert_keys_dumbbell(xy, L, R, w):
    uv = domain_to_unit_square_xy(xy, L, R, w)
    ij = np.floor(uv * nside).astype(int)
    ij = np.clip(ij, 0, nside - 1)
    return np.array([
        hc.distance_from_point([int(ix), int(iy)])
        for ix, iy in ij
    ])


def lattice_korobov(n, a, seed=None):
    """Generate the shifted second coordinate of the 2D Korobov rule."""
    n = int(n)
    a = int(a)

    if n < 4:
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
    return (((a * i) % n) / n + delta) % 1.0


def wos_dumbbell_array_rqmc(x0, w, L, R, N, qmc_type="sobol",
                            max_steps=1000, eps=1e-5, seed=None):
    x0 = np.asarray(x0, dtype=float)

    walkers = np.tile(x0, (N, 1))
    acc = np.zeros(N, dtype=float)
    active = np.ones(N, dtype=bool)
    walk_lengths = np.full(N, max_steps, dtype=int)    

    korobov_a = None
    if qmc_type == "lattice":
        korobov_a = get_korobov_a_dim2(N)

    for step in range(max_steps):
        r = np.array([
            distance_to_boundary_dumbbell(x, L, R, w)
            for x in walkers
        ])

        done_now = active & (r <= eps)
        walk_lengths[done_now] = step
        active[done_now] = False

        if not np.any(active):
            break

        step_seed = None if seed is None else seed * (max_steps + 1) + step

        if qmc_type == "sobol":
            sampler = Sobol(dimension=1, seed=step_seed, randomize=True)
            u = sampler.gen_samples(N).reshape(-1)
        elif qmc_type == "lattice":
            u = lattice_korobov(N, a=korobov_a, seed=step_seed)            
        else:
            raise ValueError(f"Unknown qmc_type: {qmc_type}")

        keys = hilbert_keys_dumbbell(walkers, L, R, w)
        keys = np.where(active, keys, np.inf)
        order = np.argsort(keys, kind="mergesort")

        assigned_u = np.empty(N)
        assigned_u[order] = u

        theta = 2.0 * np.pi * assigned_u
        to_move = active

        acc[to_move] += 0.5 * r[to_move]**2
        walkers[to_move] += r[to_move, None] * np.column_stack([
            np.cos(theta[to_move]),
            np.sin(theta[to_move])
        ])

    return np.mean(acc), np.mean(walk_lengths)


