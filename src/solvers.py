from __future__ import annotations

import math

from abc import ABC, abstractmethod
from copy import deepcopy
from heapq import heappop, heappush
from math import factorial
from random import randrange, shuffle, uniform
from p5.pmath.utils import dist


INFTY = float("inf")


class Solver(ABC):

    def __init__(self, nodes: list[tuple[int, int]]):
        self.nodes = nodes
        self.node_count = len(nodes)
        self.order = [i for i in range(self.node_count)]
        self.adj = [
            [dist(i, j) if i != j else 0 for i in nodes] for j in nodes
        ]

    @staticmethod
    def get_solver(solver_name: str, nodes: list[tuple[int, int]]) -> "Solver":
        """
        Return a new solver based off the name
        """
        solver_name = solver_name.lower().replace(" ", "")
        if solver_name == "bruteforce":
            return BruteForce(nodes)
        if solver_name == "branchandbound":
            return BranchAndBound(nodes)
        if solver_name == "simulatedannealing":
            return SimulatedAnnealing(nodes)
        raise Exception("Invalid solver name")

    def get_total_dist(self, order: list[int]) -> float:
        """
        Get the total distance between the nodes based off the ordering
        """
        if not order:
            return 0
        total = 0
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
        pass

    @abstractmethod
    def get_next_order(self) -> list[int]:
        """
        Returns the list of the next ordering of the path.
        @return an empty list if there is no more orderings.
        """
        pass

    @abstractmethod
    def get_best_order(self) -> list[int]:
        """
        @return the list of the current best ordering.
        """
        pass


################################################################################


class SimulatedAnnealing(Solver):

    def __init__(self, nodes, temperature: float = None, cooling_rate: float = None):
        super().__init__(nodes)
        shuffle(self.order)
        self.temperature = temperature if temperature else 100
        self.cooling_rate = cooling_rate if cooling_rate else 0.9999
        self.curr_dist = self.get_total_dist(self.order)
        self.iterations = 0

    def solve(self):
        repeat = 0
        order = []
        while repeat < 1000:
            # TODO: Change to use max iteration
            new_order = self.get_next_order()
            if order == new_order:
                repeat += 1
            else:
                repeat = 0
                order = new_order
        return

    def get_next_order(self) -> list[int]:
        # Lower the temperature
        self.iterations += 1

        # TODO: Tweak the temperature cooling based off number of nodes
        self.temperature = self.temperature * self.cooling_rate

        # Find new order
        a, b = self.get_two_nodes()
        loss = self.get_swap_cost(a, b)
        prob = 0 if loss <= 0 else math.exp(-loss / self.temperature)
        # print(f"{self.curr_dist = } \t{self.temperature = } \t{loss = }")

        # If new distance shorter, or within probability then use it
        if loss <= 0 or uniform(0, 1) < prob:
            self.two_opt(a, b)
            self.curr_dist = self.get_total_dist(self.order)

        return self.order

    def get_best_order(self) -> list[int]:
        return self.order

    def get_two_nodes(self) -> tuple[int, int]:
        """
        @return: two indexes between 0 and n, where the first is smaller
        """
        a = randrange(self.node_count)
        b = randrange(self.node_count)
        return (a, b) if a < b else (b, a)

    def get_swap_cost(self, a: int, b: int) -> float:
        """
        Given two indexes, return the cost if we were to reverse the
        ordering between the two indexes

        @param a: the lower index
        @param b: the higher index
        @return: the change in distance after the swap
        """
        # Get distance from a to next val
        n = self.node_count

        a1 = self.order[a]
        a2 = self.order[(a + 1) % n]
        b1 = self.order[b]
        b2 = self.order[(b + 1) % n]

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


################################################################################


class BruteForce(Solver):

    def __init__(self, nodes):
        super().__init__(nodes)
        self.best_order = [i for i in range(self.node_count)]

    def solve(self) -> None:
        for _ in range(factorial(self.node_count - 1)):
            self.get_next_order()
        return

    def get_next_order(self) -> list[int]:
        """
        Returns the next path in lexicographical ordering
        """
        order = self.order.copy()
        x = -1
        for i in range(1, self.node_count - 1):
            if order[i] < order[i + 1]:
                x = i
        if x == -1:
            return []
        y = -1
        for j in range(self.node_count):
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

    def __init__(self, nodes):
        super().__init__(nodes)
        self.adj = [
            [dist(i, j) if i != j else INFTY for i in nodes] for j in nodes
        ]
        self.cost = reduce_adj(self.adj, self.node_count)
        adj_arr = self.adj.copy()
        cost = reduce_adj(adj_arr, self.node_count)
        self.paths = [BranchAndBoundPath(adj_arr, cost, [0])]
        self.found_best = False

    def solve(self) -> None:
        order = self.get_best_order()
        while len(order) < self.node_count:
            order = self.get_best_order()
        return

    def get_next_order(self) -> list[int]:
        """
        Returns the next optimal path that we have found so far
        """
        if self.found_best:
            return []

        # Look at the best possible non complete
        curr_path = heappop(self.paths)
        curr_node = curr_path.order[-1]
        curr_adj = curr_path.adj
        curr_cost = curr_path.cost

        # List of the next possible nodes that we can go to
        next_nodes = [i for i in range(self.node_count) if i not in curr_path.order]

        # Generate their paths
        for next_node in next_nodes:
            # Find the new adj matrix after travelling to the next node
            new_adj = set_infty(curr_adj, curr_node, next_node)

            # Base cost is the cost of travelling from the curr_node to next_node
            base_cost = curr_adj[curr_node][next_node]
            new_cost = curr_cost + base_cost + reduce_adj(new_adj, self.node_count)

            new_order = curr_path.order + [next_node]

            heappush(self.paths, BranchAndBoundPath(new_adj, new_cost, new_order))
        
        result = self.get_best_order()
        self.found_best = len(result) == self.node_count
        return result

    def get_best_order(self) -> list[int]:
        order = self.paths[0].order
        return order


################################################################################


class BranchAndBoundPath:

    def __init__(self, adj: list[list[float]], cost: float, order: list[int]):
        self.adj = adj
        self.cost = cost
        self.order = order

    def __lt__(self, other) -> bool:
        return self.cost < other.cost


def set_infty(arr: list[list[float]], row_idx: int, col_idx: int) -> list[list[float]]:
    """
    Given a 2D square array and the index of a row and column, return a new list
    where the values of the row and col is now set to infinity
    """
    new_arr = deepcopy(arr)
    n = len(arr)
    for i in range(n):
        for j in range(n):
            if i == row_idx or j == col_idx:
                new_arr[i][j] = INFTY
    return new_arr


def reduce_adj(arr: list[list[float]], n: int) -> int:
    """
    Subtract the minimum value of each row from the row, and subtract the
    minimum value of each col from the col. Then return the total subtracted
    """
    total = 0

    for i in range(n):

        # Find the minimum value in the row
        row = [val for val in arr[i] if val != INFTY]

        if row:
            row_min = min(row)
            total += row_min

            # Subtract it from every value in the row
            for j in range(n):
                if arr[i][j] != INFTY:
                    arr[i][j] -= row_min

    for j in range(n):
        # Find the minimum value of the column
        col = [arr[i][j] for i in range(n) if arr[i][j] != INFTY]

        if col:
            col_min = min(col)
            total += col_min

            # Subtract it from values in column
            for i in range(n):
                if arr[i][j] != INFTY:
                    arr[i][j] -= col_min
    return total

