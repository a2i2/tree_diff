from .tree import DecisionNode, TreeMetadata
import river
from pathlib import Path
import tree_diff.tree_ruleset_conversion as conversion
from collections import deque

def nodes(tree):
    nodes = []
    tree.walk(lambda n: nodes.append(n))
    num_nodes = len(nodes)
    return num_nodes

def save_tree(tree, path):
    txt = tree.pretty_print()
    with open(f"{path}_pretty.txt", "w") as f:
        f.write(txt)

    dot = tree.plot()
    Path(f"{path}_render").mkdir(parents=True, exist_ok=True)
    dot.render(directory=f'{path}_render', view=False)


# EFDT
def river_walk_tree(node, callable):
    """Generator function that walks a tree."""
    stack = deque()
    stack.append(node)
    while stack:
        node = stack.pop()
        callable(node)
        if node.n_leaves != 1: # if not leaf
            for child in node.children:
                stack.append(child)

def river_nodes(tree_model):
    nodes = []
    river_walk_tree(tree_model._root, lambda n: nodes.append(n))
    num_nodes = len(nodes)
    return num_nodes

def river_save_tree(tree_model, path):
    with open(f"{path}_model.txt", "w") as f:
        f.write(f'{tree_model}')

    with open(f"{path}_summary.txt", "w") as f:
        f.write(f'{tree_model.summary}')

    df = tree_model.to_dataframe() # can be None (if no children)
    if df is not None:
        df.to_csv(f"{path}_df.csv")

    g = tree_model.draw()
    print(g)
    g.render(f'{path}_render', view=False)
