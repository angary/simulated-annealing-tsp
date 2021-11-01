import argparse
from typing import List
from p5 import *
from random import randint

from src.solvers import Solver

WIDTH = 680
HEIGHT = 480
BACKGROUND_COLOUR = (35, 36, 37)
BEST_PATH_COLOUR = (0, 153, 255)
CURR_PATH_COLOR = (75, 75, 75)

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
    background(*BACKGROUND_COLOUR)
    return


def draw() -> None:
    global cities, order, iteration
    # print(iteration)
    iteration += 1
    
    # Black background
    background(*BACKGROUND_COLOUR)

    # Draw cities
    fill(*BEST_PATH_COLOUR)
    for i, city in enumerate(cities):
        ellipse(city[0], city[1], 6, 6)
        # text(str(i), city[0], city[1])

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
    i = 0
    while new_order == order:
        # TODO: Loops infinitely when SA algo does not find a better soln
        # and temperature is 0
        if i == 500:
            no_loop()
            return
        new_order = solver.get_next_order()
        i += 1
    order = new_order
    if not order:
        no_loop()
        return
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
