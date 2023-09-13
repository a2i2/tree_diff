from .tree import DecisionNode, TreeMetadata
from .similar_tree import *
import river
from collections import deque

visit = {}
expected = {}


def find_to_condition(visited, nodes):
    if nodes.is_root():
        return None
    else:
        index = -1
        for i, node in enumerate(nodes.parent.children):
            if node.node_id == nodes.node_id:
                index = i
        if index < 0:
            raise ValueError("Incorrect tree")
        # print(nodes.label)
        if len(nodes.children) == 0:
            visited["{}:{}".format(node.parent, nodes)] = [
                nodes.parent.conditions[index],
                nodes.label,
            ]
        else:
            visited["{}:{}".format(node.parent, nodes)] = [
                nodes.parent.conditions[index]
            ]
        return visited


def traverse(tree, visited):
    for child in tree:
        visit = find_to_condition(visited, child)
        if len(child.children) > 0:
            visit = traverse(child.children, visited)
    return visit


def link_dict_keys(d):
    linked_dict = {}
    result = {}
    for key, value in d.items():
        if ":" in key:
            parent, child = key.split(":")
            if parent not in linked_dict:
                linked_dict[parent] = {}
            linked_dict[parent][child] = value
        else:
            linked_dict[key] = {key: value}

    for key, child_dict in linked_dict.items():
        for child, value in child_dict.items():
            if child in linked_dict:
                grandchild_dict = linked_dict[child]
                for grandchild, grandchild_value in grandchild_dict.items():
                    result[f"{key}:{child}:{grandchild}"] = value + grandchild_value

    for keys in d.keys():
        if keys not in ",".join(result.keys()):
            result[keys] = d[keys]
    return result


def tuple_tree_conversion(tree):
    visited = {}
    ruleset = []
    expected = link_dict_keys(traverse(tree.children, visited))
    for val in expected.values():
        ruleset.append(Rule(val[-1], val[0:-1]))
    return Ruleset(ruleset)


# EFDT Rule-set conversion


def walk_tree(node, fetch_children, is_leaf):
    """Generator function that walks a tree.

    fetch_children: Callable that accepts a node and path_to_node.
        Returns a List[Tuple] where Tuple = (path_to_node, child_node)

    is_leaf: Callable that accepts a node and returns true if a leaf node.

    Usage:
    >>> [p for p in walk_tree(tree, lambda x, p: [(p, x.left), (p, x.right)], lambda x: x.is_leaf)]
    """
    stack = deque()
    stack.append(([], node))
    while stack:
        path_to_node, node = stack.pop()
        if is_leaf(node):
            yield path_to_node + [node]
        else:
            children = fetch_children(node, path_to_node)
            for child in children:
                stack.append(child)


def river_children(node, path):
    if isinstance(node, river.tree.nodes.efdtc_nodes.EFDTNominalMultiwayBranch):
        return [(path + [(node, i)], n) for i, n in enumerate(node.children)]
    return [(path + [node], n) for n in node.children]


def river_is_leaf(node):
    return node.n_leaves == 1


def river_return_condition(node):
    if isinstance(node, river.tree.nodes.efdtc_nodes.EFDTNumericBinaryBranch):
        return Condition(f"attr_{node.feature}", Operator.LE, node.threshold)
    elif isinstance(node, tuple):  # Multinomial
        feature = node[0].feature
        threshold = node[0]._r_mapping[node[1]]
        return Condition(f"attr_{feature}", Operator.EQ, threshold)
    else:
        raise ValueError(node)


def river_create_conditions(path_conds):
    return [river_return_condition(c) for c in path_conds]


def river_create_rule(path):
    a = path[-1].stats
    m = (None, 0)
    for k, v in a.items():
        if not m or m[1] < v:
            m = (k, v)
    label = m[0]
    return Rule(conditions=river_create_conditions(path[0:-1]), label=f"{label}")


def river_extract_rules(tree, children, is_leaf):
    return [river_create_rule(p) for p in walk_tree(tree, children, is_leaf)]
