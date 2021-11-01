# Simulated Annealing for TSP

## Abstract

The Travelling Salesman Problem is a well known NP-Hard problem. Given a list of cities, find the shortest path that
visits all cities once.

Simulated annealing is a well known heuristic method for solving optimisation problems and is a well known non-exact
algorithm for solving the TSP. However, determining the starting temperature and cooling rate is often done empirically.

The goal of this project is to:

- Find the optimal starting temperature and cooling rate based off the input
- Visualise the solving process of the TSP

## Setup

| File / Folder | Purpose |
| --- | --- | 
| [data](data) | This contains TSP problems in `.tsp` files and their optimal solution in `.opt.tour` files, taken from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) |
| [report](report) | The report detailing the Simulated Annealing and the experimentation |
| [results](results) | The output directory containing results of the tests |
| [src/main.py](src/main.py) | Driver code to start the visualisation |
| [src/setup.py](src/setup.py) | Code for loading in city coordinates from a file, or generating random ones |
| [src/solvers.py](src/solvers.py) | Module containing the python implementations of TSP solving algorithms |

Note that this project uses the [p5py](https://github.com/p5py/p5) library for visualisation which may have some issues running on Windows.
