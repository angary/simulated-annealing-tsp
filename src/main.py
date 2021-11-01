import argparse
from typing import List
from p5 import *
from random import randint

from src.solvers import Solver
from src.config import WIDTH, HEIGHT, BG_COLOUR, BEST_PATH_COLOUR, CURR_PATH_COLOUR


city_count = 0
cities = []
order = []
solver = Solver.get_solver("simulated annealing", cities)
iteration = 0


def main() -> None:
    global city_count, cities, order, solver

    args = parse_args()
    city_count = args.city_count
    solver_name = args.solver

    cities = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1)) for _ in range(city_count)]
    order = [i for i in range(city_count)]
    solver = Solver.get_solver(solver_name, cities)

    run()
    return


def setup() -> None:
    size(WIDTH, HEIGHT)
    background(*BG_COLOUR)
    return


def draw() -> None:
    global cities, order, iteration
    iteration += 1
    background(*BG_COLOUR)

    # Draw cities
    fill(*BEST_PATH_COLOUR)
    for i, city in enumerate(cities):
        ellipse(city[0], city[1], 6, 6)

    # Draw current path
    no_fill()
    # stroke(*CURR_PATH_COLOR)
    # stroke_weight(1)
    # draw_path(order)

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


def draw_path(ordering: List[int]) -> None:
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
