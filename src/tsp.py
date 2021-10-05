from p5 import *
from random import randint, shuffle

from src.BruteForce import BruteForce

WIDTH = 680
HEIGHT = 480
CITY_COUNT = 7

cities = [(randint(0, WIDTH - 1), randint(0, HEIGHT - 1)) for _ in range(CITY_COUNT)]
order = [i for i in range(CITY_COUNT)]
best_dist = 0
best_path = []
solver = BruteForce(cities)


def main():
    global best_dist, best_path
    best_dist = total_dist()
    best_path = cities.copy()
    run()
    return


def setup():
    size(WIDTH, HEIGHT)
    background(0)
    return


def draw():
    global cities, best_dist, best_path
    
    # Black background
    background(0)

    # Draw cities
    for city in cities:
        ellipse(city[0], city[1], 4, 4)
    
    # Draw current path
    stroke(50)
    stroke_weight(1)
    no_fill()
    begin_shape()
    for city in cities:
        vertex(city[0], city[1])
    vertex(cities[0][0], cities[0][1])
    end_shape()

    # Draw best path
    stroke(255, 0, 0)
    stroke_weight(2)
    begin_shape()
    for city in best_path:
        vertex(city[0], city[1])
    vertex(best_path[0][0], best_path[0][1])
    end_shape()

    # Get next ordering
    order = solver.get_next_order()
    if order == []:
        no_loop()
        return
    
    cities = [cities[i] for i in order]

    curr_dist = total_dist()
    if curr_dist < best_dist:
        best_dist = curr_dist
        best_path = cities.copy()
    return


def total_dist() -> float:
    """
    Calculate the total distance between all the cities
    """
    total = 0
    for i in range(CITY_COUNT - 1):
        a = cities[i]
        b = cities[i + 1]
        total += dist(a, b)
    total += dist(cities[-1], cities[0])
    return total


if __name__ == "__main__":
    main()
