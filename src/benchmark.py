from __future__ import annotations

import argparse
import csv
import os

from src.setup import load_cities
from src.solvers import SimulatedAnnealing


TEST_REPEATS = 20
TEMPERATURES = [100, 1_000, 10_000, 100_000, 1_000_000]
COOLING_RATES = [0.9, 0.99, 0.999, 0.9999, 0.99999]


def main() -> None:
    args = parse_args()
    filename = args.file
    t = args.temperature
    r = args.cooling_rate

    if filename:
        # If we are given a file, print out the results
        filename = remove_file_extension(filename)
        results = benchmark(filename, t, r)
        print(filename)
        for key, val in results.items():
            print(key, val)
    else:
        # Else run tests on all the problems, and write results to results folder
        benchmark_all()
    return


def benchmark_all() -> None:
    """
    Test the solver against all the files in the data folder and combination of
    temperature and cooling rates
    """

    # Get all the problem files that have solutions
    all_files = os.listdir("data")
    data = set(map(remove_file_extension, all_files))
    data_files = [f"data/{f}" for f in data if f"{f}.opt.tour" in all_files]

    # Sort them by the number of nodes
    data_files.sort(key=lambda name: int("".join([s for s in name if s.isdigit()])))

    # Loop through each file and then test the file
    for data_file in data_files:
        problem = data_file.lstrip("data/")

        # Test what happens when temperature is 0 (doesn't matter what cooling rate is
        greedy_results = [benchmark(data_file, 0, 0) for _ in range(TEST_REPEATS)]
        write_results(problem, 0, 0, greedy_results)

        # Test for combination of temperature and cooling rates
        for t in TEMPERATURES:
            for r in COOLING_RATES:
                results = [benchmark(data_file, t, r) for _ in range(TEST_REPEATS)]
                write_results(problem, t, r, results)
    return


def write_results(problem: str, t: float, r: float, results: list[dict[str, float]]) -> str:
    """
    Write the results of a problem to a csv file

    @param problem: name of the problem
    @param t: initial temperature
    @param r: cooling rate
    @param results: list of the results
    @return: the path to the result file
    """
    result_file = f"results/{problem}_{t}_{r}.csv"
    with open(result_file, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    return result_file


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
        "optimality": soln_dist / solver_dist,
        "temperature": solver.initial_temperature,
        "avg_city_dist": solver.avg_city_dist(),
        "cooling_rate": solver.cooling_rate,
        "city_count": solver.node_count,
        "iterations": solver.iterations - solver.max_repeats,
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
        try:
            return list(map(lambda x: int(x) - 1, lines[start:end]))
        except ValueError:
            lines = [i for i in " ".join(lines[start:end]).split()]
            return list(map(lambda x: int(x) - 1, lines))


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
