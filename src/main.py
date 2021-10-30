import argparse

from src.solvers import Solver
from src.tsp import start


def main():
    """
    Run the program
    TODO: Add extra arguments, i.e.
    - City size
    - Load cities from file
    """
    start()


def get_solver_name() -> str:
    """
    Return the name of the tsp solver
    """
    args = parse_args()
    return args.solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--solver",
        type=str,
        default="simulated annealing",
        help="the type of tsp solver to use"
    )


if __name__ == "__main__":
    main()
