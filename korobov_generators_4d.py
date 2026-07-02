import json
import math
from pathlib import Path

import latnetbuilder


def find_korobov_generator(
    n: int,
    dimension: int,
    output_root: str,
):
    """
    Search for the best Korobov generating vector

        (1, a, a^2, ..., a^(dimension-1)) mod n

    using LatNet Builder's P2 criterion.
    """
    if not isinstance(n, int) or n < 4:
        raise ValueError("n must be an integer at least 4.")

    if n & (n - 1):
        raise ValueError("n must be a power of two.")

    exponent = int(math.log2(n))

    search = latnetbuilder.SearchLattice()
    search.modulus = f"2^{exponent}"
    search.construction = "ordinary"
    search.dimension = dimension
    search.exploration_method = "Korobov"
    search.figure_of_merit = "CU:P2"
    search.norm_type = "2"
    search.weights = ["product:1"]

    output_folder = Path(output_root) / f"N={n}_dim={dimension}"

    search.execute(
        output_folder=str(output_folder),
        stdout_filename="stdout.txt",
        stderr_filename="stderr.txt",
        display_progress_bar=False,
        delete_files=False,
    )

    if search.my_output is None:
        raise RuntimeError(
            f"LatNet Builder returned no output for N={n}."
        )

    result = search.my_output.result_obj

    if result is None:
        raise RuntimeError(
            f"LatNet Builder returned no result for N={n}. "
            f"Check {output_folder / 'stderr.txt'}."
        )

    vector = [int(z) for z in result.gen_vector]
    merit = float(result.merit)

    if len(vector) != dimension:
        raise RuntimeError(
            f"Expected a vector of length {dimension}, got {vector}."
        )

    a = vector[1]

    expected_vector = [
        pow(a, j, n)
        for j in range(dimension)
    ]

    if vector != expected_vector:
        raise RuntimeError(
            "The returned vector does not have Korobov form.\n"
            f"Returned: {vector}\n"
            f"Expected: {expected_vector}"
        )

    return a, vector, merit



N_list = [2**k for k in range(2, 18)]

generators = {}

for n in N_list:
    a, vector, merit = find_korobov_generator(
        n=n,
        dimension=4,
        output_root="latnetbuilder_dim4",
    )

    generators[str(n)] = {
        "a": a,
        "generating_vector": vector,
        "P2": merit,
    }

    print(
        f"N={n}: a={a}, vector={vector}, P2={merit:.8e}",
        flush=True,
    )

output_file = Path("korobov_generators_dim4.json")

with output_file.open("w") as file:
    json.dump(generators, file, indent=2)

print(f"Saved to {output_file.resolve()}")