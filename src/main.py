import argparse

from p5 import *

from src.config import WIDTH, HEIGHT, BG_COLOUR, BEST_PATH_COLOUR
from src.setup import get_random_cities, load_cities, normalise_coords
from src.solvers import Solver


city_count = 0
cities = []
order = []
solver = Solver.get_solver("simulated annealing", cities)
iteration = 0


city_drawings = None


def main() -> None:
    global city_count, cities, order, solver

    args = parse_args()
    city_count = args.city_count
    solver_name = args.solver
    filepath = args.file

    if filepath:
        print("There was a filepath specified")
        loaded_cities = load_cities(filepath)
        print(f"Loaded the cities {loaded_cities = }")
        solver = Solver.get_solver(solver_name, loaded_cities)
        cities = normalise_coords(loaded_cities, HEIGHT, WIDTH)
    else:
        cities = get_random_cities(HEIGHT, WIDTH, city_count)
        solver = Solver.get_solver(solver_name, cities)
    order = [i for i in range(city_count)]

    run()

    return


def setup() -> None:
    size(WIDTH, HEIGHT)
    background(*BG_COLOUR)
    fill(*BEST_PATH_COLOUR)
    no_fill()
    return


def draw() -> None:
    global cities, order, iteration
    iteration += 1
    background(*BG_COLOUR)

    # Can draw cities here, but very computationally intensive

    # Draw best path
    stroke(*BEST_PATH_COLOUR)
    stroke_weight(2)
    draw_path(solver.get_best_order())

    # Get next ordering
    new_order = order
    while new_order == order:
        new_order = solver.get_next_order()
    order = new_order
    if not order:
        no_loop()
    return


def draw_path(ordering: list[int]) -> None:
    """
    Draw the path of the cities

    @param ordering: list containing the order of which city to draw
    @return: None
    """
    begin_shape()
    for i in ordering:
        vertex(*cities[i])
    vertex(*cities[ordering[0]])
    end_shape()
    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--city-count",
        type=int,
        default=100,
        help="the number of cities in the problem"
    )
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="simulated annealing",
        help="the type of tsp solver to use"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="the filename of the tsp problem - if none is selected, then a random problem is generated"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
