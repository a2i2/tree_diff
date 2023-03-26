from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Callable


class Operator(str, Enum):
    EQ = "=="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    NE = "!="

class ModelType(int, Enum):
    TREE = 1
    RULE = 2
    TREE_ENSEMBLE = 3


@dataclass(frozen=True)
class Condition:
    attribute: str
    operator: Operator
    threshold: float

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.attribute} {self.operator} {self.threshold}"


@dataclass(frozen=True)
class Rule:
    label: str
    conditions: FrozenSet[Condition]

    def __getitem__(self, key):
        return self.conditions[key]

    def __str__(self):
        str_rep = " ˄ ".join([str(c) for c in self.conditions])
        str_rep += f" → {self.label}"
        return str_rep

    def __len__(self):
        return len(self.conditions)

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class Ruleset:
    rules: FrozenSet[Rule]
    tie_breaker: Callable

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, key):
        return self.rules[key]

    def __str__(self):
        return "\n".join([f"r{i}: {str(r)}" for i, r in enumerate(self.rules)])

    def __repr__(self):
        return str(self)

    def predict(self, X):
        return