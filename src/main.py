"""
Driver code for the solving visualisation
"""

from __future__ import annotations

import argparse

from p5 import *

from src.config import WIDTH, HEIGHT, BG_COLOUR, BEST_PATH_COLOUR, CITY_SIZE, BORDER
from src.setup import get_random_cities, load_cities, normalise_coords
from src.solvers import Solver


# Problem variables
city_count = 0
cities: list[tuple[float, float]] = []
order: list[int] = []
solver = Solver.get_solver("simulated annealing", cities)
iteration = 0

# Visualisation variables
paused = False
show_cities = False
speed = 1


def main() -> None:
    """
    Setup the solver, load or generate the cities, and then start the visualisation
    """
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
        cities = normalise_coords(loaded_cities, HEIGHT, WIDTH, BORDER)
    else:
        cities = get_random_cities(HEIGHT, WIDTH, city_count)
        solver = Solver.get_solver(solver_name, cities)
    order = list(range(city_count))
    run()


def setup() -> None:
    """
    Set up the size of the window and background colour
    """
    size(WIDTH, HEIGHT)
    background(*BG_COLOUR)
    no_fill()


def draw() -> None:
    """
    Draw out each frame
    """
    global order, iteration
    iteration += 1
    background(*BG_COLOUR)

    # Draw best path
    stroke(*BEST_PATH_COLOUR)
    stroke_weight(2)
    draw_path(solver.get_best_order())

    # Can draw cities here, but very computationally intensive
    if show_cities:
        draw_cities()
    # Speed up drawing
    for _ in range(city_count * speed):
        solver.get_next_order()
    # Get next ordering
    new_order = order
    while new_order == order:
        new_order = solver.get_next_order()
    order = new_order
    if not order:
        no_loop()


def key_pressed(event) -> None:
    """
    Handle key press events
    On space-bar press it toggles pause
    On "c" press, it toggles showing the cities
    On an arrow press, it reduces or increases the frequency of the drawing

    @param event: the keypress event
    """
    global paused, show_cities, speed
    if event.key == " ":
        global paused
        if not paused:
            no_loop()
        else:
            loop()
        paused = not paused
    elif event.key == "c":
        # TODO: Show cities if it is paused
        global show_cities
        show_cities = not show_cities
    elif event.key == "LEFT":
        # Halve the speed
        speed = max(1, speed // 2)
        print(f"{speed = }")
    elif event.key == "RIGHT":
        # Double the speed
        speed = min(256, speed * 2)
        print(f"{speed = }")


def draw_cities() -> None:
    """
    Given the cities, draw then out on the screen
    """
    fill(*BEST_PATH_COLOUR)
    for x, y in cities:
        ellipse(x, y, CITY_SIZE, CITY_SIZE)
    no_fill()


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


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    @return an object containing parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--city-count",
        type=int, default=100,
        help="the number of cities in the problem"
    )
    parser.add_argument(
        "-s", "--solver",
        type=str, default="simulated annealing",
        help="the type of tsp solver to use"
    )
    parser.add_argument(
        "-f", "--file",
        type=str, default=None,
        help="the tsp filename - if none is selected, then generate a random problem"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
