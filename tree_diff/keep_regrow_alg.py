import numpy as np
import pandas as pd
from collections import Counter
import sklearn
import sklearn.tree
import uuid
from .conversion import sklearn_to_tree
from .cost_tree import CostNode, CostMetadata, to_cost_tree
from .tree import DecisionNode, count_values, gini_impurity
from .tree import grow_tree as our_grow_tree

OLD = False # Old node (is_new = False)
NEW = True  # New node (is_new = True)


# utility function
def _uid():
    return uuid.uuid4().int


def sklearn_grow_func(X, y, max_depth, metadata):
    # Sklearn implementation of tree growing
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf = clf.fit(X, y)
    return sklearn_to_tree(clf, metadata.column_names)

def tree_grow_func(X, y, max_depth, metadata):
    # Our implementation of tree growing
    return our_grow_tree(pd.DataFrame(X, columns=metadata.column_names), y, max_depth=max_depth)


def grow_tree(X, y, old_tree=None, max_depth=4, alpha=1, beta=0.25, grow_func=tree_grow_func, **kwargs):
    # alpha controls penalty for complexity (number of nodes)
    # beta controls additional penalty for changes (number of new nodes)
    column_names = X.columns
    X = X.to_numpy()
    classes = list(np.unique(y))

    if old_tree is None:
        old_tree = CostNode(best_pred(y, classes), 0, [])

    if not isinstance(old_tree, CostNode):
        assert isinstance(old_tree, DecisionNode)
        old_tree = to_cost_tree(old_tree)

    metadata = CostMetadata(classes, column_names, alpha, beta, grow_func)
    
    flag_old(old_tree)
    recost(X, y, old_tree, metadata)
    
    tree = reduce(X, y, old_tree, max_depth, metadata)
    return tree


def filter_data(X, y, node=None):
    if node is None:
        return X, y
    
    matches = node.matches(X)
    return X[matches], y[matches]


def best_pred(y, classes):
    # Adapted frm https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
    # Including one element of each of class at start ensures result even if empty (and breaks ties)
    data = Counter(classes + list(y))
    return data.most_common(1)[0][0]


def regrow(X, y, node, max_depth, metadata):
    # TODO: Option to switch between sklearn and our own implementation
    _X, _y = filter_data(X, y, node)
    max_depth = max_depth - node.depth
    
    if max_depth <= 0:
        # leaf node (sklearn cannot generate tree with depth 0)
        tree = DecisionNode(None, 0, [])
    else:
        # sklearn splits are non-deterministic unless we set random_state
        tree = metadata.grow_func(_X, _y, max_depth, metadata)

    tree = to_cost_tree(tree)
    return tree


def flag_old(node):
    # Flags all nodes in tree as old
    # is responsibility of caller to call recost after
    node.is_new = OLD

    for child in node.children:
        flag_old(child)


def reeval(X, y, node, metadata, fix_label=False):
    _, _y = filter_data(X, y, node)
    if fix_label:
        node.label = best_pred(_y, metadata.classes)
        print(f"fixing label {node}")
        
    node.value = count_values(_y, metadata.classes)
    node.impurity = gini_impurity(_y)


def recost(X, y, node, metadata, fix_label=False):
    reeval(X, y, node, metadata, fix_label)

    # visit children first so information can propogate bottom up
    for child in node.children:
        recost(X, y, child, metadata, fix_label)

    node.recost(metadata)


def prune(X, y, node, metadata):
    _, _y = filter_data(X, y, node)
    pred = best_pred(_y, metadata.classes)
    value = count_values(_y, metadata.classes)
    impurity = gini_impurity(_y)

    prune_tree = CostNode(pred, _uid(), value, impurity)
    prune_tree.recost(metadata)
    
    if node.is_leaf():
        # Keeping existing leaf node is never better than prune_tree.
        # Unlike existing leaf node, prune_tree always uses best pred label.
        keep_tree = prune_tree
    else:
        keep_tree = CostNode(pred, _uid(), value, impurity)
        new_children = [prune(X, y, child, metadata) for child in node.children]
        for child, condition in zip(new_children, node.conditions):
            keep_tree.add_child(condition, child)
        keep_tree.recost(metadata)

    if keep_tree.total_cost <= prune_tree.total_cost:
        return keep_tree
    return prune_tree


def reduce(X, y, node, max_depth, metadata):
    # TODO: consider re-evaluating node.value, node.impurity here rather than requiring reeval'ed tree
    if node.is_leaf():
        keep_tree = CostNode(
            node.label, node.node_id, node.value, node.impurity,
            None, [], [], OLD, node.total_cost
        )
    else:
        new_children = [reduce(X, y, child, max_depth, metadata) for child in node.children]
        # We do not pay changed node penality if retaining structure of old tree.
        keep_tree = CostNode(
            node.label, node.node_id, node.value, node.impurity,
            None, [], [], OLD, node.total_cost
        )
        for child, condition in zip(new_children, node.conditions):
            keep_tree.add_child(condition, child)
        keep_tree.recost(metadata)

    regrow_tree = regrow(X, y, node, max_depth, metadata)
    _X, _y = filter_data(X, y, node) # needed to ensure that we can filter data to this node even though regrow_tree parent not set
    regrow_tree = prune(_X, _y, regrow_tree, metadata)

    if keep_tree.total_cost <= regrow_tree.total_cost:
        return keep_tree
    return regrow_tree
