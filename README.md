# Paramterised Simulated Annealing for TSP

<p align="center">
  <img src="examples/world-tsp.gif" alt="animated" />
</p>


## Abstract

The Travelling Salesman Problem is a well known NP-Hard problem. Given a list of cities, find the shortest path that
visits all cities once.

Simulated annealing is a well known heuristic method for solving optimisation problems and is a well known non-exact
algorithm for solving the TSP. However, determining the starting temperature and cooling rate is often done empirically.

The goal of this project is to:

- Parameterising the optimal starting temperature and cooling rate based off the input
- Visualise the solving process of the TSP

## Usage

### Running the code

Examples of common commands to run the files are shown below.
However, both src/main.py and src/benchmark.py have a `--help` that explains the optional flags.

```sh
# To visualise annealing on a problem set from the input file
python3 -m src.main -f <input_file>

# To visualise TSP on a random graph with <city_count> number of cities
python3 -m src.main -c <city_count>

# Benchmark the parameters using all problems in the data folder
python3 -m src.benchmark
```

### Creating your own model

If you would like to create your own instance of the TSP problem and visualise it:

1. Create a new file
2. Within this file ensure you have the line `NODE_COORD_SECTION`, and below that `EOF`.
3. Between those two lines, you can place the coordinates of the cities, i.e. for the nth city, have a line like `<n> <x> <y>`, where `x` and `y` are the x and y coordinates of the city.
4. Run `python3 -m src.main -f <file>`, where `<file>` is the path to the file you have just made.

## Files

| File / Folder                        | Purpose                                                                                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| [data](data)                         | This contains TSP problems in `.tsp` files and their optimal solution in `.opt.tour` files, taken from TSPLIB |
| [report](report)                     | The report detailing the Simulated Annealing and the experimentation                                          |
| [results](results)                   | The output directory containing results of the tests                                                          |
| [src/benchmark.py](src/benchmark.py) | Code for benchmarking different temperatures and cooling rates using the problems in the data folder          |
| [src/main.py](src/main.py)           | Driver code to start the visualisation                                                                        |
| [src/setup.py](src/setup.py)         | Code for loading in city coordinates from a file, or generating random ones                                   |
| [src/solvers.py](src/solvers.py)     | Module containing the python implementations of TSP solving algorithms                                        |

Note that this project uses the [p5py](https://github.com/p5py/p5) library for visualisation which may have some issues running on Windows.
