import numpy as np
from dataclasses import dataclass, field
from .tree import DecisionNode

MISCLASSIFICATION_COST = 1
NODE_COST = 1 + 0.25 # complexity cost + audit cost


# utility function
def _indent(s):
    return '\n'.join("    " + line for line in s.split('\n'))


@dataclass
class CostNode(DecisionNode):
    internal_cost: float = NODE_COST
    total_cost: float = 0.0

    def recost(self, classes):
        # complexity cost + audit cost
        self.total_cost = self.internal_cost
        
        # cost of misclassification
        if self.is_leaf():
            if self.label not in classes:
                misclassifications = sum(self.value)
            else:
                misclassifications = sum(self.value) - self.value[self.label_index(classes)]
            self.total_cost += misclassifications * MISCLASSIFICATION_COST
        
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
                f"LEAF {self.find_to_condition()} (Cost: {self.total_cost}, Label: {self.label}, Values: {self.value})"
            )

        # non-leaf node
        s = f"NODE {'ROOT' if self.is_root() else self.find_to_condition()} (Cost: {self.total_cost})\n"
        for child in self.children:
            s += _indent(child.pretty_print()) + '\n'
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
