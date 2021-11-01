import argparse

from typing import List

from src.setup import load_cities
from src.solvers import SimulatedAnnealing


def main() -> None:
    args = parse_args()
    filename = args.file
    t = args.temperature
    r = args.cooling_rate

    # Find file names
    prob_file = f"{filename}.tsp"
    soln_file = f"{filename}.opt.tour"

    # Load in the cities and the solution
    loaded_cities = load_cities(prob_file)
    soln_order = load_soln(soln_file)

    solver = SimulatedAnnealing(loaded_cities, temperature=t, cooling_rate=r)
    solver.solve()
    solver_order = solver.get_best_order()

    print(len(soln_order))
    print(len(solver_order))

    soln_dist = solver.get_total_dist(soln_order)
    solver_dist = solver.get_total_dist(solver_order)
    optimality = soln_dist / solver_dist
    print(f"{soln_dist = }")
    print(f"{solver_dist = }")
    print(f"{optimality = }")
    return


def load_soln(filepath: str) -> List[int]:
    """
    Loads and returns the optimal solution from the given file path

    @param filepath: the filepath to the .opt.tour file
    @return: the list of the optimal ordering
    """
    with open(filepath) as f:
        lines = f.readlines()
        start = lines.index("TOUR_SECTION\n") + 1
        end = lines.index("-1\n")
        return list(map(lambda x: int(x) - 1, lines[start:end]))


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    @return: object containing file argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="the filename of the tsp problem without the file extension"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=int,
        default=None,
        help="the starting temperature of the system"
    )
    parser.add_argument(
        "-r",
        "--cooling-rate",
        type=int,
        default=None,
        help="the cooling rate of the system"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()