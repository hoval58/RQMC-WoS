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
    """Returns the precomputed dimension-2 Korobov multiplier for n."""
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


# ===== Constants ========================================
FLT_MAX = np.finfo(np.float32).max
M_PI = np.pi
maxStepsPerWalk = 32
  

# ===== Scene Data =======================================

# Temperature values (°C)
Touter = 120.0   #boundary conditions
Tcoolant = 90.0
Toil = 130.0
Toilreturn = 110.0
Tbore = 160.0

# ----- Circles -----
circles = [
    # vec2(x, y), radius, boundary value
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
    ((0.465414, -0.367129), 0.006136, Toil)
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
    ((0.003010, 0.382168), (0.458203, 0.382168), Touter)
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
    ((-0.458346, -0.318667), 0.047240, 2.483835, 4.967462, Toilreturn)
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
    qx =  c * p0[:, 0] - s * p0[:, 1]
    qy =  s * p0[:, 0] + c * p0[:, 1]
    qx = np.abs(qx)
    sc = np.array([np.sin(phi), np.cos(phi)])
    # per-point branching
    cond = sc[1] * qx > sc[0] * qy
    dist = np.where(
        cond,
        np.linalg.norm(np.stack([qx, qy], axis=1) - sc * radius, axis=1),
        np.abs(np.sqrt(qx**2 + qy**2) - radius)
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
        di = distanceCircle_vec(x, C)
        if di < dmin:
            dmin = di
            g = C[2]
    for S in segments:
        di = distanceSegment_vec(x, S)
        if di < dmin:
            dmin = di
            g = S[2]
    for A in arcs:
        di = distanceArc_vec(x, A)
        if di < dmin:
            dmin = di
            g = A[4]
    return g
    
# ====== Walk-on-Spheres solver ===========================

def wos_mc(x0, N=100, seed=None, eps=0.001):
    np.random.seed(seed)
    max_steps = maxStepsPerWalk

    # Pre-generate all random angles
    points = np.random.uniform(0, 2*np.pi, size=(N, max_steps))
  
    results = []
    cpt_list = []

    for i in range(N):
        x = np.array(x0, dtype=float)
        step = 0  
        while step < max_steps:
            r = distanceBoundary_vec(x)
            if r <= eps:
                cpt_list.append(step)
                break
            theta = points[i, step]
            x += r * np.array([np.cos(theta), np.sin(theta)])
            step += 1
        
        results.append(boundaryValue(x))

    return np.mean(results), np.mean(cpt_list)

def wos_gasket_rqmc(x0, N=100, type="sobol", seed=None,eps=0.001): #standard WOS
    """
    Walk-on-Spheres in the gasket using RQMC points for N trajectories.

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
    cpt_list = []
    
    max_steps = maxStepsPerWalk
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
        x = np.array(x0, dtype=float)
        step = 0  
        while step < max_steps:
            r = distanceBoundary_vec(x)
            if r <= eps:
                cpt_list.append(step)
                break
            theta = 2*np.pi*points[i, step]
            x += r * np.array([np.cos(theta), np.sin(theta)])
            step += 1
        
        results.append(boundaryValue(x))

    return np.mean(results), np.mean(cpt_list)

def disk_to_unit_square_xy(xy: np.ndarray) -> np.ndarray:
    """
    returns a (n,2) array with points in the unit disk
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
    u = (((a * i) % n) / n + delta) % 1.0

    return u




def wos_gasket_array_rqmc(x0, N, qmc_type="sobol", max_steps=1000, eps=1e-3, seed=None):
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

        step_seed = None if seed is None else seed * (max_steps + 1) + step

        # --- Fresh RQMC points for this step ---
        if qmc_type == "sobol":
            sampler = Sobol(dimension=1, seed=step_seed, randomize=True)
            u = sampler.gen_samples(N).reshape(-1)   # (N,)
        elif qmc_type == "halton":
            sampler = Halton(dimension=1, seed=step_seed, randomize=True)
            u = sampler.gen_samples(N).reshape(-1)
        elif qmc_type == "lattice":
            u = lattice_korobov(N, a=korobov_a, seed=step_seed)
        else:
            raise ValueError(f"Unknown qmc_type: {qmc_type}")

        # --- Sort all walkers by their Hilbertkey; push inactive walkers to end ---
        keys = hilbert_keys_hilbertcurve(walkers)
        keys = np.where(active, keys, np.inf)
        order = np.argsort(keys, kind="mergesort")

        assigned_u = np.empty(N)
        assigned_u[order] = u

        # --- Move only active walkers not already within eps ---
        r = distanceBoundary_vec(walkers)
        to_move = active & (r > eps)

        theta = 2.0 * np.pi * assigned_u
        walkers[to_move] += r[to_move, None] * np.stack(
            [np.cos(theta[to_move]), np.sin(theta[to_move])], axis=1
        )

        # --- Mark as inactive the newly done ---
        r_new = distanceBoundary_vec(walkers)
        newly_done = active & (r_new <= eps)
        walk_lengths[newly_done] = step + 1
        active[newly_done] = False

    results = np.array([boundaryValue(walkers[i]) for i in range(N)])
    return np.mean(results), np.mean(walk_lengths)


