from typing import List
from p5 import *
from random import randint

from src.solvers import Solver

WIDTH = 680
HEIGHT = 480
BACKGROUND_COLOUR = (35, 36, 37)
BEST_PATH_COLOUR = (0, 153, 255)
CURR_PATH_COLOR = (75, 75, 75)

CITY_COUNT = 100
cities = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1)) for _ in range(CITY_COUNT)]
order = [i for i in range(CITY_COUNT)]
solver = Solver.get_solver("simulated annealing", cities)

iteration = 0


def start():
    run()
    return


def setup():
    size(WIDTH, HEIGHT)
    background(*BACKGROUND_COLOUR)
    return


def draw():
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
    while new_order == order:
        # TODO: Loops infinitely when SA algo does not find a better soln
        # and temperature is 0
        new_order = solver.get_next_order()
    order = new_order
    if not order:
        no_loop()
        return
    return


def draw_path(ordering: List[int]) -> None:
    """
    Given the cities, draw the path of the cities by plotting the vertices
    """
    begin_shape()
    for i in ordering:
        vertex(*cities[i])
    vertex(*cities[ordering[0]])
    end_shape()
    return


if __name__ == "__main__":
    start()
