from abc import ABC, abstractmethod
from heapq import heappop, heappush
from typing import List, Tuple
from p5.pmath.utils import dist


INFTY = float("inf")


class Solver(ABC):

    def __init__(self, nodes: List[Tuple[int, int]]):
        self.nodes = nodes
        self.node_count = len(nodes)
        self.order = [i for i in range(self.node_count)]
        self.best_order = [i for i in range(self.node_count)]

    @staticmethod
    def get_solver(solver_name: str, nodes: List[Tuple[int, int]]) -> "Solver":
        """
        Return a new solver based off the name
        """
        solver_name = solver_name.lower().replace(" ", "")
        if solver_name == "bruteforce":
            return BruteForce(nodes)
        if solver_name == "branchandbound":
            return BranchAndBound(nodes)
        raise Exception("Invalid solver name")

    def get_total_dist(self, order: List[int]) -> float:
        """
        Get the total distance between the nodes based off the ordering
        """
        total = 0
        nodes = self.nodes
        for i in range(len(order) - 1):
            a = nodes[order[i]]
            b = nodes[order[i + 1]]
            total += dist(a, b)
        total += dist(nodes[order[-1]], nodes[order[0]])
        return total

    @abstractmethod
    def get_next_order(self) -> List[int]:
        """
        Returns the list of the next ordering of the path.
        Returns an empty list if there is no more orderings.
        """
        pass

    @abstractmethod
    def get_best_order(self) -> List[int]:
        """
        Return the list of the current best ordering.
        """
        pass


################################################################################


class BruteForce(Solver):

    def __init__(self, nodes):
        super().__init__(nodes)

    def get_next_order(self) -> List[int]:
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

        print(self.order)
        return self.order

    def get_best_order(self) -> List[int]:
        return self.best_order


################################################################################

class BranchAndBound(Solver):

    def __init__(self, nodes):
        super().__init__(nodes)
        self.order = [0]
        self.best_order = [0]
        self.adj = [
            [dist(i, j) if i != j else INFTY for i in nodes] for j in nodes
        ]
        self.cost = reduce_adj(self.adj, self.node_count)

        adj_arr = self.adj.copy()
        cost = reduce_adj(adj_arr, self.node_count)
        self.paths = [BranchAndBoundPath(adj_arr, cost, [0])]

        self.found_best = False

    def get_next_order(self) -> List[int]:
        """
        Returns the next optimal path that we have found so far
        """
        if self.found_best:
            return []

        # Look at the best possible non complete
        curr_path = self.paths.pop(0)
        curr_node = curr_path.order[-1]
        curr_adj = curr_path.adj
        curr_cost = curr_path.cost

        # List of the next possible nodes that we can go to
        next_nodes = [i for i in range(self.node_count) if i not in curr_path.order]

        # Generate their paths
        for next_node in next_nodes:
            # Find the new adj matrix after travelling to the next node
            new_adj = [[j for j in i] for i in curr_adj]
            new_adj = set_infty(new_adj, curr_node, next_node)

            # Base cost is the cost of travelling from the curr_node to next_node
            base_cost = curr_adj[curr_node][next_node]
            new_cost = curr_cost + base_cost + reduce_adj(new_adj, self.node_count)
            new_order = curr_path.order + [next_node]

            heappush(self.paths, BranchAndBoundPath(new_adj, new_cost, new_order))

        result = self.get_best_order()
        self.found_best = len(result) == self.node_count
        return result

    def get_best_order(self) -> List[int]:
        order = self.paths[0].order
        return order


class BranchAndBoundPath:

    def __init__(self, adj: List[List[float]], cost: float, order: List[int]):
        self.adj = adj
        self.cost = cost
        self.order = order

    def __lt__(self, other) -> bool:
        return self.cost < other.cost


################################################################################

def set_infty(arr: List[List[float]], row_idx: int, col_idx: int) -> List[List[float]]:
    """
    Given a 2D square array and the index of a row and column, return a new list
    where the values of the row and col is now set to infinity
    """
    new_arr = [[j for j in i] for i in arr]
    n = len(arr)
    for i in range(n):
        for j in range(n):
            if i == row_idx or j == col_idx:
                new_arr[i][j] = INFTY
    return new_arr


def reduce_adj(arr: List[List[float]], n: int) -> int:
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
