import argparse
import os

from src.setup import load_cities
from src.solvers import SimulatedAnnealing


def main() -> None:
    args = parse_args()
    filename = args.file
    t = args.temperature
    r = args.cooling_rate

    if filename:
        filename = remove_file_extension(filename)
        results = {
            filename: benchmark(filename, t, r)
        }
    else:
        results = benchmark_all(t, r)

    for file, results in results.items():
        print(file)
        for key, val in results.items():
            print(key, val)

    return


def benchmark_all(t: int, r: int) -> dict[str, dict[str, float]]:
    """
    Test the solver against all the files in the data folder

    @param t: the temperature
    @param r: the cooling rate
    @return: a dictionary mapping from filename to test results
    """

    # Get all the problem files
    data_files = [os.path.join("data", f) for f in os.listdir("data") if ".tsp" in f]

    # Sort them by the number of nodes
    data_files.sort(key=lambda name: int("".join([s for s in name if s.isdigit()])))

    # Loop through each file and then test the file
    results = {}
    for file in data_files:
        file = remove_file_extension(file)
        results[file] = benchmark(file, t, r)

    return results


def benchmark(filename: str, t: int, r: int) -> dict[str, float]:
    """
    Benchmark the simulated annealing solver against a dataset

    @param filename: the name of the tsp problem data
    @param t: the temperature
    @param r: the cooling rate
    @return: a dictionary containing test results
    """

    # Find file names
    prob_file = f"{filename}.tsp"
    soln_file = f"{filename}.opt.tour"

    # Load in the cities and the solution
    loaded_cities = load_cities(prob_file)
    soln_order = load_soln(soln_file)

    # Get the solver's solution
    solver = SimulatedAnnealing(loaded_cities, temperature=t, cooling_rate=r)
    solver.solve()
    solver_order = solver.get_best_order()

    soln_dist = solver.get_total_dist(soln_order)
    solver_dist = solver.get_total_dist(solver_order)

    # Return a dictionary containing test results
    return {
        "soln_dist": soln_dist,
        "solver_dist": solver_dist,
        "optimality": soln_dist / solver_dist
    }


def load_soln(filepath: str) -> list[int]:
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


def remove_file_extension(filename: str) -> str:
    """
    @param filename: the filename to remove the extension from
    @return: the filename without the file extension
    """
    if "." in filename:
        filename = filename[:filename.index(".")]
    return filename


def parse_args() -> argparse.Namespace:
    """
    Parse commandline arguments

    @return: object containing commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="the filename of the tsp problem - if nothing is specified all files are tested"
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
