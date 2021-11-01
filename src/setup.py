from random import randint
from typing import List, Tuple


def get_random_cities(height: int, width: int, n: int) -> List[Tuple[int, int]]:
    """
    Generate a series of random coordinates within the bounds of the
    height and width

    :param height: the maximum height
    :param width: the maximum width
    :param n: the number of coordinates
    :return: a list of city coordinates
    """
    cities = []
    for _ in range(n):
        cities.append((randint(0, width - 1), randint(0, height - 1)))
    return cities


def load_cities(filepath: str) -> List[Tuple[float, float]]:
    """
    Load the cities in from a file, returning a list of the cities coordinates

    :param filepath: the filepath to the file containing the tsp data
    :return: a list of the city coordinates
    """
    cities = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        start_idx = lines.index("NODE_COORD_SECTION\n")
        lines = lines[start_idx + 1:-1]
        for line in lines:
            cities.append(list(map(float, line.split()[1:])))
    return cities


def normalise_coords(cities: List[Tuple[float, float]], height: int, width: int) -> List[Tuple[int, int]]:
    """
    Given the list of cities, return a new list, which ensures that the
    coordinates are all positive and scale their values so they all lie
    within the height and width. Used for ensuring that the city
    coordinates can be visualised on a screen.

    :param cities: the list of city coordinates
    :param height: the maximum height
    :param width: the maximum width
    :return: a list of normalised city coordinates
    """
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Find scaling factor
    scale_x = (width - 1) / (max_x + min_x)
    scale_y = (height - 1) / (max_y + min_y)
    normalised = []
    for c in cities:

        # Ensure that all the cities are properly scaled
        normalised_x = int(scale_x * (c[0] + min_x))
        normalised_y = height - int(scale_y * (c[1] + min_y))
        normalised.append((normalised_x, normalised_y))
    return normalised
