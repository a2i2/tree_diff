import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from .tree import DecisionNode, TreeMetadata


# utility function
def _indent(s):
    return '\n'.join("    " + line for line in s.split('\n'))

@dataclass
class CostMetadata(TreeMetadata):
    alpha: float
    beta: float
    grow_func: Callable[[np.ndarray, np.ndarray, int, TreeMetadata], DecisionNode]

@dataclass
class CostNode(DecisionNode):
    is_new: bool = True
    total_cost: float = 0.0

    def recost(self, metadata):
        # complexity cost
        self.total_cost = metadata.alpha

        # audit cost
        if self.is_new:
            self.total_cost += metadata.beta

        # cost of misclassification
        if self.is_leaf():
            if self.label not in metadata.classes:
                correct = 0
            else:
                correct = self.value[self.label_index(metadata.classes)]
            
            incorrect = sum(self.value) - correct
            self.total_cost += incorrect

        for child in self.children:
            self.total_cost += child.total_cost

    def matches_row(self, x):
        node = self
        
        while True:
            cond = node.find_to_condition()
            if cond is None:
                # Reached the root node and all conditions in parent chain held
                return True
            if not cond.fire(x):
                # Condition in parent chain did not hold
                return False
            node = node.parent

    def matches(self, X):
        # TODO: Define on DecisionNode class instead
        return np.apply_along_axis(self.matches_row, 1, X)

    def pretty_print(self):
        # TODO: Reimplement
        if self.is_leaf():
            return (
                f"{'NEW' if self.is_new else 'OLD'} LEAF {self.find_to_condition()} (Cost: {self.total_cost}, Label: {self.label}, Values: {self.value})"
            )

        # non-leaf node
        s = f"{'NEW' if self.is_new else 'OLD'} NODE {'ROOT' if self.is_root() else self.find_to_condition()} (Cost: {self.total_cost})\n"
        s += '\n'.join(_indent(child.pretty_print()) for child in self.children)
        return s

    @property
    def depth(self):
        # root node is 0 depth
        depth = 0
        p = self.parent
        while p is not None:
            p = p.parent
            depth += 1
        return depth


def to_cost_tree(node):
    new_node = CostNode(
        node.label,
        node.node_id,
        node.value,
        node.impurity
    )

    for child, condition in zip(node.children, node.conditions):
        new_child = to_cost_tree(child)
        new_node.add_child(condition, new_child)
    return new_node
