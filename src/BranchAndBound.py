from typing import List
from src.Solver import Solver


class BranchAndBound(Solver):

    def __init__(self, nodes):
        super().__init__(nodes)


    def get_next_order(self) -> List[int]:
        """
        Returns the next path in lexicographical ordering
        """
        order = self.order
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
        self.order = order[:x + 1] + order[:x:-1]
        return order
