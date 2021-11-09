from __future__ import annotations

from random import randint


def get_random_cities(height: int, width: int, n: int) -> list[tuple[int, int]]:
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
        cities.append((randint(0, width - 1), randint(0, height - 1)))
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
            coords.reverse()
            cities.append(coords)
    return cities


def normalise_coords(cities: list[tuple[float, float]], height: int, width: int, border: int) -> list[tuple[int, int]]:
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
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]

    # Find minimum and maximum values
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Convert minimum values to positive if needed
    min_x = max(-min_x, 0)
    min_y = max(-min_y, 0)

    # Find scaling factor
    scale_x = (width - border * 2) / (max_x + min_x)
    scale_y = (height - border * 2) / (max_y + min_y)

    # Choose the smaller scale to prevent overflow
    scale = min(scale_x, scale_y)

    normalised = []
    for c in cities:
        # Ensure that all the cities are properly scaled
        normalised_x = int(scale * (c[0] + min_x)) + border
        normalised_y = height - int(scale * (c[1] + min_y)) - border
        normalised.append((normalised_x, normalised_y))
    return normalised
