"""
Module containing class implementations of different TSP solver algorithms.
The classes extend an abstract Solver class, which contains abstract method
solve(), get_next_order(), and get_best_order()
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from heapq import heappop, heappush
from math import dist, factorial
from random import randrange, shuffle, uniform

INFTY = float("inf")


class Solver(ABC):
    """
    Abstract class to be extended by other solvers
    """
    def __init__(self, cities: list[tuple[int, int]]):
        self.cities = cities
        self.n = len(cities)
        self.order = list(range(self.n))
        self.adj = [
            [dist(i, j) if i != j else 0 for i in cities] for j in cities
        ]

    @staticmethod
    def get_solver(solver_name: str, cities: list[tuple[float, float]]) -> "Solver":
        """
        Return a new solver based off the name

        @param solver_name: the name of the solver
        @param cities: a list of the coordinates of the cities
        @return: the solver with the respective name
        """
        solver_name = solver_name.lower().replace(" ", "")
        if solver_name == "bruteforce":
            return BruteForce(cities)
        if solver_name == "branchandbound":
            return BranchAndBound(cities)
        if solver_name == "simulatedannealing":
            return SimulatedAnnealing(cities)
        raise Exception("Invalid solver name")

    def get_total_dist(self, order: list[int]) -> float:
        """
        Get the total distance between the cities based off the ordering

        @param order: a list containing the order of cities to visit
        @return: the total distance of travelling in the order and returning to the start
        """
        if not order:
            return 0
        total = 0.0
        for i in range(len(order) - 1):
            total += self.adj[order[i + 1]][order[i]]
        total += self.adj[order[-1]][order[0]]
        return total

    @abstractmethod
    def solve(self) -> None:
        """
        Loop the get_next_order() method until the solver
        has found the optimal (or what it determines) to be
        the optimal solution
        """

    @abstractmethod
    def get_next_order(self) -> list[int]:
        """
        Returns the list of the next ordering of the path.
        @return an empty list if there is no more orderings.
        """

    @abstractmethod
    def get_best_order(self) -> list[int]:
        """
        @return the list of the current best ordering.
        """


################################################################################


class SimulatedAnnealing(Solver):
    """
    Solver using the simulated annealing algorithm
    """

    def __init__(self, cities, temperature: float = 100, cooling_rate: float = 0.999):
        super().__init__(cities)
        shuffle(self.order)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.initial_temperature = self.temperature
        self.curr_dist = self.get_total_dist(self.order)
        self.solved = False
        self.__iterations = 0
        self.__max_repeats = int(10 * (1 / (1 - self.cooling_rate)))
        self.__acceptance = []

    def solve(self) -> None:
        """
        Continue cooling and finding distance until the optimal distance has
        not changed after self.max_repeats iterations
        """
        repeat = 0
        lowest_dist = float("inf")
        while repeat < self.__max_repeats:
            self.get_next_order()
            if self.curr_dist < lowest_dist:
                repeat = 0
                lowest_dist = self.curr_dist
            else:
                repeat += 1
        self.solved = True

    def get_next_order(self) -> list[int]:
        # Lower the temperature
        self.__iterations += 1

        self.temperature = self.temperature * self.cooling_rate

        # Find new order
        a, b = self.get_two_cities()
        loss = self.get_swap_cost(a, b)
        prob = 0 if (loss <= 0 or self.temperature <= 0) else math.exp(-loss / self.temperature)

        # If new distance shorter, or within probability then use it
        if loss <= 0:
            self.two_opt(a, b)
            self.curr_dist = self.get_total_dist(self.order)
        else:
            if uniform(0, 1) < prob:
                self.__acceptance.append(True)
                self.two_opt(a, b)
                self.curr_dist = self.get_total_dist(self.order)
            else:
                self.__acceptance.append(False)
        return self.order

    def get_best_order(self) -> list[int]:
        return self.order

    def get_two_cities(self) -> tuple[int, int]:
        """
        @return: two indexes between 0 and n, where the first is smaller
        """
        a = randrange(self.n)
        b = randrange(self.n)
        return (a, b) if a < b else (b, a)

    def get_swap_cost(self, a: int, b: int) -> float:
        """
        Given two indexes, return the cost if we were to reverse the
        ordering between the two indexes

        @param a: the lower index
        @param b: the higher index
        @return: the change in distance after the swap
        """
        n, order = self.n, self.order

        # Find which cities a and b are, and their next city
        a1, a2 = order[a], order[(a + 1) % n]
        b1, b2 = order[b], order[(b + 1) % n]

        # Find the current and new distance
        curr_dist = self.adj[a1][a2] + self.adj[b1][b2]
        new_dist = self.adj[a1][b1] + self.adj[a2][b2]

        return new_dist - curr_dist

    def two_opt(self, a: int, b: int) -> None:
        """
        Reverse the position between two values in the ordering,
        so that the path "uncrosses" itself

        @param a: the lower index
        @param b: the higher index
        """
        self.order = self.order[:a + 1] + self.order[b:a:-1] + self.order[b + 1:]

    @property
    def iterations(self) -> int:
        """
        @return the current number of iterations, else if solved return the number
                of iterations required to converge to the solution
        """
        if self.solved:
            return self.__iterations - self.__max_repeats
        return self.__iterations

    @property
    def acceptance_ratio(self) -> float:
        """
        @return the current acceptance ratio of the solver else if solved, return
                the acceptance ratio before it could not find a better solution
        """
        end = 1 if not self.solved else self.__max_repeats
        acceptances = self.__acceptance[:-end]
        return acceptances.count(True) / len(acceptances)


################################################################################


class BruteForce(Solver):
    """
    Solver by brute forcing the lexicographical ordering
    """

    def __init__(self, cities):
        super().__init__(cities)
        self.best_order = list(range(self.n))

    def solve(self) -> None:
        for _ in range(factorial(self.n - 1)):
            self.get_next_order()

    def get_next_order(self) -> list[int]:
        """
        Find the next path in lexicographical ordering

        @return: the next path found
        """
        order = self.order.copy()
        x = -1
        for i in range(1, self.n - 1):
            if order[i] < order[i + 1]:
                x = i
        if x == -1:
            return []
        y = -1
        for j in range(self.n):
            if order[x] < order[j]:
                y = j
        order[x], order[y] = order[y], order[x]
        order = order[:x + 1] + order[:x:-1]

        # Update ordering
        self.order = order

        # Find the best ordering
        curr_dist = self.get_total_dist(self.best_order)
        new_dist = self.get_total_dist(self.order)
        if new_dist <= curr_dist:
            self.best_order = order

        return self.order

    def get_best_order(self) -> list[int]:
        return self.best_order


################################################################################


class BranchAndBound(Solver):
    """
    Solver using the branch and bound algorithm
    """

    def __init__(self, cities):
        super().__init__(cities)
        self.adj = [
            [dist(i, j) if i != j else INFTY for i in cities] for j in cities
        ]
        self.cost = reduce_adj(self.adj, self.n)
        adj_arr = self.adj.copy()
        cost = reduce_adj(adj_arr, self.n)
        self.paths = [BranchAndBoundPath(adj_arr, cost, [0])]
        self.found_best = False

    def solve(self) -> None:
        order = self.get_best_order()
        while len(order) < self.n:
            order = self.get_best_order()

    def get_next_order(self) -> list[int]:
        """
        Find the next order using the branch and bound method

        @return: the next optimal path that we have found so far
        """
        if self.found_best:
            return []

        # Look at the best possible non complete
        curr_path = heappop(self.paths)
        curr_node = curr_path.order[-1]
        curr_adj = curr_path.adj
        curr_cost = curr_path.cost

        # List of the next possible cities that we can go to
        next_cities = [i for i in range(self.n) if i not in curr_path.order]

        # Generate their paths
        for next_node in next_cities:
            # Find the new adj matrix after travelling to the next node
            new_adj = set_infty(curr_adj, curr_node, next_node)

            # Base cost is the cost of travelling from the curr_node to next_node
            base_cost = curr_adj[curr_node][next_node]
            new_cost = curr_cost + base_cost + reduce_adj(new_adj, self.n)

            new_order = curr_path.order + [next_node]
            heappush(self.paths, BranchAndBoundPath(new_adj, new_cost, new_order))

        result = self.get_best_order()
        self.found_best = len(result) == self.n
        return result

    def get_best_order(self) -> list[int]:
        order = self.paths[0].order
        return order


################################################################################


class BranchAndBoundPath:
    """
    A class for which contains an adjacency matrix, cost and ordering which
    has an __lt__ method to support sorting by cost
    """

    def __init__(self, adj: list[list[float]], cost: float, order: list[int]):
        self.adj = adj
        self.cost = cost
        self.order = order

    def __lt__(self, other) -> bool:
        return self.cost < other.cost


def set_infty(arr: list[list[float]], r: int, c: int) -> list[list[float]]:
    """
    Given a 2D square array and the index of a row and column, return a new list
    where the values of the row and col is now set to infinity

    @param arr: 2d list of floats
    @param r: the row to set to infinity
    @param c: the col to set to infinity
    @return: the new list with values set to infinity
    """
    n = len(arr)
    return [[INFTY if (i == r or j == c) else arr[i][j] for j in range(n)] for i in range(n)]


def reduce_adj(arr: list[list[float]], n: int) -> float:
    """
    Subtract the minimum value of each row from the row, and subtract the
    minimum value of each col from the col. Then return the total subtracted

    @param arr: 2d list of floats
    @return: the total value reduced
    """
    total = 0.0

    for i in range(n):
        row = [val for val in arr[i] if val != INFTY]
        if row:
            row_min = min(row)
            total += row_min
            for j in range(n):
                if arr[i][j] != INFTY:
                    arr[i][j] -= row_min
    for j in range(n):
        col = [arr[i][j] for i in range(n) if arr[i][j] != INFTY]
        if col:
            col_min = min(col)
            total += col_min
            for i in range(n):
                if arr[i][j] != INFTY:
                    arr[i][j] -= col_min
    return total
