import numpy as np
from collections import Counter
import sklearn
import sklearn.tree
import uuid
from .conversion import sklearn_to_tree
from .cost_tree import CostNode, to_cost_tree
from .tree import DecisionNode, TreeMetadata, count_values, gini_impurity

MAX_DEPTH = 4
OLD_NODE_COST = 1 # complexity cost only


# util functions:
def uid():
    return uuid.uuid4().int

def grow_tree(X, y, old_tree=None, max_depth=MAX_DEPTH, **kwargs):
    column_names = X.columns
    X = X.to_numpy()
    classes = list(np.unique(y))

    if old_tree is None:
        old_tree = CostNode(best_pred(y, classes), 0, [])

    if not isinstance(old_tree, CostNode):
        assert isinstance(old_tree, DecisionNode)
        old_tree = to_cost_tree(old_tree)

    metadata = TreeMetadata(classes, column_names)
    
    adjust_internal_cost(old_tree, OLD_NODE_COST)
    recost(X, y, old_tree, metadata)
    
    # print("DEBUG A")
    # print(old_tree.pretty_print())
    # print("DEBUG A END")
    
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
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        clf = clf.fit(_X, _y)
        tree = sklearn_to_tree(clf, metadata.column_names)

    tree = to_cost_tree(tree)
    return tree

def adjust_internal_cost(node, node_cost):
    node.internal_cost = node_cost

    for child in node.children:
        adjust_internal_cost(child, node_cost)
    
    # is responsibility of caller to call recost after

def reeval(X, y, node, metadata, fix_label=False):
    
    # def update(n):
    #     _, _y = filter_data(X, y, n)
    #     node.value = count_values(_y, metadata.classes)
    #     node.impurity = gini_impurity(_y)
    # 
    # node.walk(update)

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

    node.recost(metadata.classes)

def prune(X, y, node, metadata):
    _, _y = filter_data(X, y, node)
    pred = best_pred(_y, metadata.classes)
    value = count_values(_y, metadata.classes)
    impurity = gini_impurity(_y)

    prune_tree = CostNode(pred, uid(), value, impurity)
    prune_tree.recost(metadata.classes)
    
    if node.is_leaf():
        # Keeping existing leaf node is never better than prune_tree.
        # Unlike existing leaf node, prune_tree always uses best pred label.
        keep_tree = prune_tree
    else:
        keep_tree = CostNode(pred, uid(), value, impurity)
        new_children = [prune(X, y, child, metadata) for child in node.children]
        for child, condition in zip(new_children, node.conditions):
            keep_tree.add_child(condition, child)
        keep_tree.recost(metadata.classes)

    if keep_tree.total_cost <= prune_tree.total_cost:
        return keep_tree
    return prune_tree

def reduce(X, y, node, max_depth, metadata):
    # TODO: consider re-evaluating node.value, node.impurity here rather than requiring reeval'ed tree
    if node.is_leaf():
        keep_tree = CostNode(
            node.label, node.node_id, node.value, node.impurity,
            None, [], [], node.internal_cost, node.total_cost
        )
    else:
        new_children = [reduce(X, y, child, max_depth, metadata) for child in node.children]
        # retaining node.internal_cost ensures we do not pay changed node penality if retaining structure of old tree.
        keep_tree = CostNode(
            node.label, node.node_id, node.value, node.impurity,
            None, [], [], node.internal_cost, node.total_cost
        )
        for child, condition in zip(new_children, node.conditions):
            keep_tree.add_child(condition, child)
        keep_tree.recost(metadata.classes)

    regrow_tree = regrow(X, y, node, max_depth, metadata)
    _X, _y = filter_data(X, y, node) # needed to ensure that we can filter data to this node even though regrow_tree parent not set
    regrow_tree = prune(_X, _y, regrow_tree, metadata)

    if keep_tree.total_cost <= regrow_tree.total_cost:
        return keep_tree
    return regrow_tree
