"""
Module containing variables used in visualisation and testing
"""

# Visualisation variables
WIDTH = 1280
HEIGHT = 720
BORDER = 40
BG_COLOUR = (35, 36, 37)
BEST_PATH_COLOUR = (0, 153, 255)
CURR_PATH_COLOUR = (75, 75, 75)
CITY_SIZE = 4

# Variables for benchmarking - stored here so it is accessible by ipynb files
MAP_COUNT = 3
TEST_REPEATS = 20
DIST_DIFFS = [1, 10, 100, 1_000, 10_000]
CITY_COUNTS = [10, 30, 100, 300, 1_000, 3_000]
CONST_CITY_COUNT = 200
CONST_DIST_DIFF = 20

# For Random Dataset testing
TEMPERATURES = [0, 1, 10, 100, 1_000, 10_000]
COOLING_RATES = [0, 0.9, 0.99, 0.999, 0.999_9, 0.999_99]
CONST_TEMPERATURE = 40
CONST_COOLING_RATE = 0.9999
