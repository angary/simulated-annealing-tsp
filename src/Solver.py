from abc import ABC, abstractclassmethod
from typing import List

from p5.pmath.utils import dist

class Solver(ABC):

    def __init__(self, nodes):
        self.node_count = len(nodes)
        self.order = [i for i in range(self.node_count)]
        self.adj_matrix = [[dist(i, j) if i != j else -1 for i in nodes] for j in nodes]


    @abstractclassmethod
    def get_next_order() -> List[int]:
        """
        Returns the list of the next ordering of the path.
        Returns an empty list if there is no more orderings.
        """
        pass
