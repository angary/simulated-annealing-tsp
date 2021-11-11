from __future__ import annotations

from math import dist
from random import randint


def get_random_cities(height: int, width: int, n: int) -> list[tuple[float, float]]:
    """
    Generate a series of random coordinates within the bounds of the
    height and width

    @param height: the maximum height
    @param width: the maximum width
    @param n: the number of coordinates
    @return: a list of city coordinates
    """
    cities = []
    for _ in range(n):
        cities.append((float(randint(0, width - 1)), float(randint(0, height - 1))))
    return cities


def load_cities(filepath: str) -> list[tuple[float, float]]:
    """
    Load the cities in from a file, returning a list of the cities coordinates

    @param filepath: the filepath to the file containing the tsp data
    @return: a list of the city coordinates
    """
    cities = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        start = lines.index("NODE_COORD_SECTION\n") + 1
        end = lines.index("EOF\n")
        lines = lines[start:end]
        for line in lines:
            coords = list(map(float, line.split()[1:3]))
            cities.append((coords[1], coords[0]))
    return cities


def normalise_coords(
    cities: list[tuple[float, float]],
    height: int,
    width: int,
    border: int
) -> list[tuple[float, float]]:
    """
    Given the list of cities, return a new list, which ensures that the
    coordinates are all positive and scale their values so they all lie
    within the height and width. Used for ensuring that the city
    coordinates can be visualised on a screen.

    @param cities: the list of city coordinates
    @param height: the maximum height
    @param width: the maximum width
    @return: a list of normalised city coordinates
    """
    xs, ys = tuple(zip(*cities))

    # Find minimum and maximum values
    min_x = abs(min(xs))
    min_y = abs(min(ys))

    # Find scaling factor
    scale_x = (width - border * 2) / (max(xs) + min_x)
    scale_y = (height - border * 2) / (max(ys) + min_y)

    # Choose the smaller scale to prevent overflow
    scale = min(scale_x, scale_y)

    normalised = []
    for c in cities:
        # Ensure that all the cities are properly scaled
        normalised_x = (scale * (c[0] + min_x)) + border
        normalised_y = height - scale * (c[1] + min_y) - border
        normalised.append((normalised_x, normalised_y))
    return normalised


def avg_city_dist(cities: list[tuple[int, int]]) -> float:
    """
    Find the sum of the paths between all the cities and then
    divide it by the number of cities

    @param cities: a list of the coordinates of the cities
    @return: the average distance between all the cities
    """
    n = len(cities)
    d = [dist(cities[i], cities[j]) for i in range(n - 1) for j in range(i + 1, n)]
    return sum(d) / len(d)


def get_diff_city_dist(cities: list[tuple[float, float]]) -> float:
    """
    Get the distances between all the cities, and then return the
    average difference of the distances

    @param cities: a list of the coordinates of the cities
    @return: the average difference in distances between the cities
    """
    n = len(cities)

    # Generate all combination of distances in sorted order in O(n^2 log n)
    d = sorted([dist(cities[i], cities[j]) for i in range(n - 1) for j in range(i + 1, n)])
    m = len(d)

    # Find the total difference between the smallest distance and the rest
    prev_diff = sum([x - d[0] for x in d[1:]])

    # Loop over the rest of the values calculating total difference
    total_diff = prev_diff
    for i in range(1, m):
        prev_diff = prev_diff - (d[i] - d[i - 1]) * (m - i)
        total_diff += prev_diff

    # Return the average distance
    return total_diff / ((m * (m - 1)) / 2)
