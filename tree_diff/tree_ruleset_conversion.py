from .tree import DecisionNode, TreeMetadata
from .similar_tree import *
import river
from collections import deque

expected = {}


def find_to_condition(visited, nodes):
    """
    This function finds the conditions associated with a given node in a tree and stores them in a dictionary."
    """
    if nodes.is_root():
        return None
    else:
        index = -1
        for i, node in enumerate(nodes.parent.children):
            if node.node_id == nodes.node_id:
                index = i
        if index < 0:
            raise ValueError("Incorrect tree")
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
    visit = {}
    for child in tree:
        visit = find_to_condition(visited, child)
        if len(child.children) > 0:
            visit = traverse(child.children, visited)
    return visit


def link_dict_keys(d):
    """
    This function takes the tree nodes and it's children to create a link between all of it's children.
    It combines the conditions of the parent, child, and grandchild nodes into a single value.
    Returns: A dictionary where keys are in the format "parent:child:grandchild" and 
    values are the sum of the conditions of the parent, child, and grandchild nodes.
    """
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
    if len(tree.children) == 0: # Check if it's a root node
        attr_name = 'root_node_tree'
        visited["root"] = [f"{attr_name} <= 0", tree.label]
        antecedent = visited["root"][0]
        ruleset.append(Rule(visited['root'][1],[f"{antecedent}"])) # Force root node to have a (antecedent) Rule and label
        return Ruleset(ruleset)
    else:
        expected = link_dict_keys(traverse(tree.children, visited)) # Traverse for each children and find linkage
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


def river_return_condition(node,path,val_sum):
    if isinstance(node, river.tree.nodes.efdtc_nodes.EFDTNumericBinaryBranch):
        weight_value = {}
        for elements in range(len(path)):
            current = []
            for key, value in path[elements].stats.items():
                current.append(value)
            weight_value[elements] = sum(current) # Store the current path status
        all_values = list(weight_value.values())
        all_values.append(val_sum)
        left,right = node.children
        if left.total_weight in all_values: # Examine if either of children's weight matches with parent weight
            operator = Operator.LE
        elif right.total_weight in all_values:
            operator = Operator.GT    
        return Condition(f"attr_{node.feature}", operator, node.threshold)
    
    elif isinstance(node, tuple):  # Multinomial
        feature = node[0].feature
        threshold = node[0]._r_mapping[node[1]]
        return Condition(f"attr_{feature}", Operator.EQ, threshold)
    
    elif isinstance(node, river.tree.nodes.efdtc_nodes.NumericBinaryBranch):
        weight_value = {}
        for elements in range(len(path)):
            current = []
            keys_index = []
            for key, value in path[elements].stats.items():
                current.append(value)
                keys_index.append(key)
            weight_value[elements] = sum(current) # Store the current path status
        all_values = list(weight_value.values())
        all_values.append(val_sum)
        left,right = node.children
        if left.total_weight in all_values: # Examine if either of children's weight matches with parent weight
            operator = Operator.LE
        elif right.total_weight in all_values:
            operator = Operator.GT    
        else:
            for elements in range(len(path)):
                if left == path[elements]:
                    operator = Operator.LE
                elif right == path[elements]:
                    operator = Operator.GT 
        return Condition(f"attr_{node.feature}", operator, node.threshold)
    else:
        raise ValueError(node)


def river_create_conditions(path_conds,val_sum):
    return [river_return_condition(c, path_conds, val_sum) for i, c in enumerate(path_conds)]


def river_create_rule(path):
    a = path[-1].stats
    weight = []
    for key,values in a.items(): # Storing the original weight of parent node
        weight.append(values)
    val_sum = sum(weight)
    m = (None, 0)
    for k, v in a.items():
        if not m or m[1] < v:
            m = (k, v)
    label = m[0]
    if len(path) == 1:  # Check if the node is a root node
        random_attr = "rand"
        return Rule(conditions=[Condition(f"attr_{random_attr}", Operator.LE, 0)], label=f"{label}")
    else:
        return Rule(conditions=river_create_conditions(path[0:-1],val_sum), label=f"{label}")


def river_extract_rules(tree, children, is_leaf):
    return [river_create_rule(p) for p in walk_tree(tree, children, is_leaf)]
