from __future__ import annotations

import argparse
import csv
import os
import random

from src.config import \
    TSPLIB_TEST_REPEATS, TSPLIB_TEMPERATURES, TSPLIB_COOLING_RATES, \
    RAND_MAP_SIZES, RAND_CITY_COUNTS, \
    RAND_CONST_MAP_SIZE, RAND_CONST_CITY_COUNT, \
    RAND_TEMPERATURES, RAND_TEST_REPEATS, RAND_COOLING_RATES, \
    RAND_CONST_TEMPERATURE, RAND_CONST_COOLING_RATE
from src.setup import get_random_cities, load_cities
from src.solvers import SimulatedAnnealing


def main() -> None:
    args = parse_args()
    gen_rand_data = args.gen_rand_data
    filename = args.file
    t = args.temperature
    r = args.cooling_rate

    if gen_rand_data:
        # Generate random maps
        random.seed(args.seed)
        gen_rand_cities()
        benchmark_all(True)
    elif filename:
        # If we are given a file, print out the results
        filename = remove_file_extension(filename)
        results = benchmark(filename, t, r)
        print(filename)
        for key, val in results.items():
            print(key, val)
    else:
        # Else run tests on all the problems, and write results to results folder
        benchmark_all(False)
    return


def gen_rand_cities() -> None:
    """
    Generate a series of random city maps
    """
    avg_count = RAND_CONST_CITY_COUNT
    avg_size = RAND_CONST_MAP_SIZE

    # Generate random maps with different city count
    for city_count in RAND_CITY_COUNTS:
        cities = get_random_cities(avg_size, avg_size, city_count)
        save_cities_into_file(cities, avg_size)

    # Generate random maps with different sizes
    for size in RAND_MAP_SIZES:
        cities = get_random_cities(size, size, avg_count)
        save_cities_into_file(cities, size)

    return


def save_cities_into_file(cities: list[tuple[int, int]], size: int) -> str:
    n = len(cities)
    name = f"rand{size}_{n}"
    filename = f"data/{name}.tsp"
    with open(filename, "w+") as f:
        lines = [
            f"NAME : {name}\n",
            f"COMMENT : randomly generated map\n",
            f"TYPE : TSP\n",
            f"DIMENSION : {n}\n",
            f"EDGE_WEIGHT_TYPE : EUC_2D\n",
            f"NODE_COORD_SECTION\n"
        ]
        f.writelines(lines)
        for i, city in enumerate(cities):
            f.write(f"{i + 1}\t{city[0]}\t{city[1]}\n")
        f.write("EOF\n")


def benchmark_all(rand: bool) -> None:
    """
    Test the solver against all the files in the data folder and combination of
    temperature and cooling rates

    @param rand: if we are testing random datasets
    """

    # Get all the problem files that have solutions
    all_files = os.listdir("data")
    data = set(map(remove_file_extension, all_files))

    # Filter problems by tsplib or random
    if rand:
        data_files = [f"data/{f}" for f in data if f.startswith("rand")]
        test_repeats, temperatures, cooling_rates = RAND_TEST_REPEATS, RAND_TEMPERATURES, RAND_COOLING_RATES
    else:
        data_files = [f"data/{f}" for f in data if f"{f}.opt.tour" in all_files]
        test_repeats, temperatures, cooling_rates = TSPLIB_TEST_REPEATS, TSPLIB_TEMPERATURES, TSPLIB_COOLING_RATES

    # Sort them by the number of nodes
    data_files.sort(key=lambda name: int("".join([s for s in name if s.isdigit()])))

    # Loop through each file and then test the file
    if not rand:
        for data_file in data_files:
            problem = data_file.removeprefix("data/")

            # Test what happens when temperature is 0 (doesn't matter what cooling rate is
            greedy_results = [benchmark(data_file, 0, 0) for _ in range(test_repeats)]
            print(write_results(problem, 0, 0, greedy_results))

            # Test for combination of temperature and cooling rates
            for t in temperatures:
                for r in cooling_rates:
                    results = [benchmark(data_file, t, r) for _ in range(test_repeats)]
                    print(write_results(problem, t, r, results))
    else:
        avg_cr = RAND_CONST_COOLING_RATE
        avg_t = RAND_CONST_TEMPERATURE
        avg_size = RAND_CONST_MAP_SIZE
        avg_count = RAND_CONST_CITY_COUNT

        # Test the different temperatures
        avg_size_files = []
        for data_file in data_files:
            size = int(data_file.split("_")[0].removeprefix("data/rand"))
            if size == avg_size:
                avg_size_files.append(data_file)

        for t in temperatures:
            for data_file in avg_size_files:
                problem = data_file.removeprefix("data/")
                print(f"testing {problem} with {t = } {avg_cr = }")
                results = [run_test(data_file, t, avg_cr) for _ in range(test_repeats)]
                print(write_results(problem, t, avg_cr, results))

        # Test the different cooling rates
        avg_count_files = []
        for data_file in data_files:
            count = int(data_file.split("_")[1])
            if count == avg_count:
                avg_count_files.append(data_file)
        for cr in cooling_rates:
            for data_file in avg_count_files:
                problem = data_file.removeprefix("data/")
                results = [run_test(data_file, avg_t, cr) for _ in range(test_repeats)]
                print(write_results(problem, avg_t, cr, results))
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


def run_test(filename: str, t: int, r: int) -> dict[str, float]:
    """
    Run test on a problem without comparing to the solution

    @param filename: the name of the tsp problem data
    @param t: the temperature
    @param r: the cooling rate
    @return: a dictionary containing test results
    """
    # Find problem and cities
    prob_file = f"{filename}.tsp"
    loaded_cities = load_cities(prob_file)

    # Get solver's solution
    solver = SimulatedAnnealing(loaded_cities, temperature=t, cooling_rate=r)
    solver.solve()
    solver_order = solver.get_best_order()
    solver_dist = solver.get_total_dist(solver_order)

    # Return a dictionary containing test results
    return {
        "solver_dist": solver_dist,
        "temperature": solver.initial_temperature,
        "avg_city_dist": solver.avg_city_dist(),
        "cooling_rate": solver.cooling_rate,
        "city_count": solver.node_count,
        "iterations": solver.iterations - solver.max_repeats,
    }


def benchmark(filename: str, t: int, r: int) -> dict[str, float]:
    """
    Benchmark the simulated annealing solver against a problem

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
    parser.add_argument(
        "-g",
        "--gen-rand-data",
        action="store_true",
        help="test the algorithm on randomly generated city datasets"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="the seed for generating random maps"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
