# Variables for the visualisation
WIDTH = 1280
HEIGHT = 720
BORDER = 40
BG_COLOUR = (35, 36, 37)
BEST_PATH_COLOUR = (0, 153, 255)
CURR_PATH_COLOUR = (75, 75, 75)
CITY_SIZE = 4

# Variables for the benchmarking (stored here so it can be accessed by ipynb)
TSPLIB_TEST_REPEATS = 20
TSPLIB_TEMPERATURES = [10, 50, 100, 500, 1_000, 5_000]
TSPLIB_COOLING_RATES = [0.999, 0.999_5, 0.999_9, 0.999_95]

# Variables for generating random cities for testing
RAND_MAP_SIZES = [50, 100, 500, 1000, 1500]
RAND_CITY_COUNTS = [50, 100, 500, 1000, 1500]
RAND_CONST_CITY_COUNT = 100
RAND_CONST_MAP_SIZE = 100

# Variables for benchmarking
RAND_TEST_REPEATS = 20
RAND_TEMPERATURES = [10, 50, 100, 500, 1000]
RAND_COOLING_RATES = [0.999, 0.9995, 0.999_9, 0.999_95, 0.999_99]
RAND_CONST_TEMPERATURE = 10
RAND_CONST_COOLING_RATE = 0.999
