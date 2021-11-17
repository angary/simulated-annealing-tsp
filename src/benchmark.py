"""
Contains code to benchmark the SimulatedAnnealing solver against TSPLIB instances
or randomly generated TSP problems
"""

from __future__ import annotations

import argparse
import csv
import os
import random

from src.config import \
    TEST_REPEATS, TEMPERATURES, COOLING_RATES, \
    CITY_COUNTS, DIST_DIFFS, MAP_COUNT, \
    CONST_CITY_COUNT, CONST_DIST_DIFF, \
    CONST_TEMPERATURE, CONST_COOLING_RATE
from src.setup import get_diff_city_dist, get_random_cities, load_cities
from src.solvers import SimulatedAnnealing


def main() -> None:
    """
    Parse command line arguments and test SA solver on random cities, all the
    TSPLIB instances, or one TSP instance
    """
    args = parse_args()
    gen_rand_data = args.gen_rand_data
    filename = args.file
    t = args.temperature
    r = args.cooling_rate
    test_temp = args.test_temperature or args.test_temperature == args.test_cooling_rate
    test_cool = args.test_cooling_rate or args.test_temperature == args.test_cooling_rate

    if gen_rand_data:
        # Generate random maps
        random.seed(args.seed)
        files = gen_rand_cities(test_temp, test_cool)
        # benchmark_rand(files)
    elif filename:
        # If we are given a file, print out the results
        filename = remove_file_extension(filename)
        results = benchmark(filename, t, r)
        print(filename)
        for key, val in results.items():
            print(key, val)
    else:
        # Else run tests on all the problems, and write results to results folder
        benchmark_tsplib()


def gen_rand_cities(test_temp: bool, test_cool: bool) -> dict[str, list[str]]:
    """
    Generate a series of random city maps

    @param test_temp: if we create temperature test files
    @param test_cool: if we create cooling rate test files
    @return: dictionary containing file paths to the cities
    """

    # Generate random maps with different city count
    # where they have the same size
    cooling_rate_tests = []
    if test_cool:
        for city_count in CITY_COUNTS:
            for i in range(MAP_COUNT):
                # Generate random points and find their average distance
                cities = get_random_cities(1000, 1000, city_count)

                # For every city scale their coordinate so they have correct avg dist
                scale = CONST_DIST_DIFF / get_diff_city_dist(cities)
                cities = [(i * scale, j * scale) for (i, j) in cities]

                filepath = save_cities_into_file(
                    cities, CONST_DIST_DIFF, i, "random cooling rate test"
                )
                cooling_rate_tests.append(filepath)
                print("Cooling rate test: ", filepath)

    # Generate random maps with different average distances between the cities
    # where they have the same average city distance
    if test_temp:
        cities_list = [
            get_random_cities(1000, 1000, CONST_CITY_COUNT) for _ in range(MAP_COUNT)
        ]
        temperature_tests = []
        for rand_city_dist in DIST_DIFFS:
            for i, cities in enumerate(cities_list):
                # Scale the city positions to the desired avg city distance
                scale = rand_city_dist / get_diff_city_dist(cities)
                scaled_cities = [(i * scale, j * scale) for (i, j) in cities]

                filepath = save_cities_into_file(
                    scaled_cities, rand_city_dist, i, "random temperature test"
                )
                temperature_tests.append(filepath)
                print("Temperature test:  ", filepath)

    return {
        "temperature_tests": temperature_tests,
        "cooling_rate_tests": cooling_rate_tests
    }


def save_cities_into_file(
    cities: list[tuple[float, float]],
    size: int,
    map_num: int,
    comment: str = ""
) -> str:
    """
    Given a list of cities, save the data into a file in TSPLIB format

    @param cities: A list of the coordinates of the cities
    @param size: the average difference in distance between all the cities
    @param map_num: integer representing the number of the map with same size and city counts
    @param comment: the comment to add in the file
    @return: the path to the file
    """
    n = len(cities)
    name = f"rand{n}_{size}_{map_num}"
    filename = f"data/{name}.tsp"
    with open(filename, "w+") as f:
        f.writelines([
            f"NAME : {name}\n",
            f"COMMENT : {comment}\n",
            "TYPE : TSP\n",
            f"DIMENSION : {n}\n",
            "EDGE_WEIGHT_TYPE : EUC_2D\n",
            "NODE_COORD_SECTION\n"
        ])
        for i, city in enumerate(cities):
            f.write(f"{i + 1}\t{city[0]}\t{city[1]}\n")
        f.write("EOF\n")
    return filename


def benchmark_rand(files: dict[str, list[str]]) -> None:
    """
    Test the solver against the randomly generated city sets

    @param files: dictionary containing list of file paths for the different problems
    """

    for data_file in files["temperature_tests"]:
        for t in TEMPERATURES:
            results = [run_test(data_file, t, CONST_COOLING_RATE) for _ in range(TEST_REPEATS)]
            print(write_results(data_file.removeprefix("data/"), t, CONST_COOLING_RATE, results))

    for data_file in files["cooling_rate_tests"]:
        for cr in COOLING_RATES:
            results = [run_test(data_file, CONST_TEMPERATURE, cr) for _ in range(TEST_REPEATS)]
            print(write_results(data_file.removeprefix("data/"), CONST_TEMPERATURE, cr, results))


def benchmark_tsplib() -> None:
    """
    Test the solver against all the tsplib problems in the data folder
    temperature and cooling rates
    """

    # Get all the problem files that have solutions
    all_files = os.listdir("data")
    data = set(map(remove_file_extension, all_files))

    # Only select problems that have an optimal tour, and sort by num cities
    data_files = [f"data/{f}" for f in data if f"{f}.opt.tour" in all_files]
    data_files.sort(key=lambda name: int("".join([s for s in name if s.isdigit()])))

    # Loop through each file and then test the file
    for data_file in data_files:
        problem = data_file.removeprefix("data/").removesuffix(".tsp")

        # Test what happens when temperature is 0 (doesn't matter what cooling rate is
        greedy_results = [benchmark(data_file, 0, 0) for _ in range(TEST_REPEATS)]
        print(write_results(problem, 0, 0, greedy_results))

        # Test for combination of temperature and cooling rates
        for t in TEMPERATURES:
            for r in COOLING_RATES:
                results = [benchmark(data_file, t, r) for _ in range(TEST_REPEATS)]
                print(write_results(problem, t, r, results))


def write_results(problem: str, t: float, r: float, results: list[dict[str, float]]) -> str:
    """
    Write the results of a problem to a csv file

    @param problem: name of the problem
    @param t: initial temperature
    @param r: cooling rate
    @param results: list of the results
    @return: the path to the result file
    """
    problem = problem.replace(".tsp", "")
    result_file = f"results/{problem}_{t}_{r}.csv"
    with open(result_file, "w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    return result_file


def run_test(filename: str, t: int, r: float) -> dict[str, float]:
    """
    Run test on a problem without comparing to the solution

    @param filename: the name of the tsp problem data
    @param t: the temperature
    @param r: the cooling rate
    @return: a dictionary containing test results
    """
    # Find problem and cities
    loaded_cities = load_cities(filename)
    avg_dist_diff = int("".join([c for c in filename.split("_")[1] if c.isdigit()]))
    map_num = int("".join([c for c in filename.split("_")[2] if c.isdigit()]))

    # Get solver's solution
    solver = SimulatedAnnealing(loaded_cities, temperature=t, cooling_rate=r)
    solver.solve()
    solver_order = solver.get_best_order()
    solver_dist = solver.get_total_dist(solver_order)

    # Return a dictionary containing test results
    return {
        "distance": solver_dist,
        "temperature": solver.initial_temperature,
        "avg_dist_diff": avg_dist_diff,
        "cooling_rate": solver.cooling_rate,
        "city_count": solver.n,
        "iterations": solver.iterations,
        "map_num": map_num,
    }


def benchmark(filename: str, t: int, r: float) -> dict[str, float]:
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
        "distance": solver_dist,
        "optimality": soln_dist / solver_dist,
        "temperature": solver.initial_temperature,
        "avg_dist_diff": get_diff_city_dist(loaded_cities),
        "cooling_rate": solver.cooling_rate,
        "city_count": solver.n,
        "iterations": solver.iterations,
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
            lines = list(" ".join(lines[start:end]).split())
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
        "-f", "--file",
        type=str, default=None,
        help="the filename of the tsp problem - if nothing is specified all files are tested"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=int, default=None,
        help="the starting temperature of the system"
    )
    parser.add_argument(
        "-r", "--cooling-rate",
        type=int, default=None,
        help="the cooling rate of the system"
    )
    parser.add_argument(
        "-g", "--gen-rand-data",
        action="store_true",
        help="test the algorithm on randomly generated city datasets"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int, default=1,
        help="the seed for generating random maps"
    )
    parser.add_argument(
        "--test-temperature",
        action="store_true",
        help="only benchmark different temperatures"
    )
    parser.add_argument(
        "--test-cooling-rate",
        action="store_true",
        help="only benchmark different cooling rates"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
