from typing import List
from p5 import *
from random import randint

from src.solvers import Solver

WIDTH = 680
HEIGHT = 480
CITY_COUNT = 10

cities = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1)) for _ in range(CITY_COUNT)]
order = [i for i in range(CITY_COUNT)]
solver = Solver.get_solver("branch and bound", cities)


def main():
    run()
    return


def setup():
    size(WIDTH, HEIGHT)
    background(0)
    return


def draw():
    global cities,order
    
    # Black background
    background(0)

    # Draw cities
    for i, city in enumerate(cities):
        ellipse(city[0], city[1], 4, 4)
        text(str(i), city[0], city[1])
    
    # Draw current path
    no_fill()
    stroke(75)
    stroke_weight(1)
    draw_path(order)

    # Draw best path
    stroke(255, 0, 0)
    stroke_weight(2)
    draw_path(solver.get_best_order())

    # Get next ordering
    order = solver.get_next_order()
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
    main()
