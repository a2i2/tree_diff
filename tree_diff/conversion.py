import numpy as np

from .tree import DecisionNode, Condition, Operator


def sklearn_to_tree(sklearn_tree, column_names):
    # Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    n_nodes = sklearn_tree.tree_.node_count
    children_left = sklearn_tree.tree_.children_left
    children_right = sklearn_tree.tree_.children_right
    feature = sklearn_tree.tree_.feature
    threshold = sklearn_tree.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    # our tree
    node_map = {} # node_id -> node

    # pass 1 (construct nodes)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))

            # Create Node
            new_node = DecisionNode(None, node_id, [])
            node_map[node_id] = new_node
        else:
            is_leaves[node_id] = True

            # Create Leaf Node
            new_node = DecisionNode(None, node_id, [])
            node_map[node_id] = new_node

    # pass 2 (link nodes)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))

            # Link children
            node = node_map[node_id]
            left = node_map[children_left[node_id]]
            right = node_map[children_right[node_id]]

            attribute_pos = feature[node_id]
            attribute = column_names[attribute_pos]
            thresh = threshold[node_id]
            
            left_cond = Condition(
                attribute,
                attribute_pos,
                Operator.LE,
                thresh
            )
            
            right_cond = Condition(
                attribute,
                attribute_pos,
                Operator.GT,
                thresh
            )

            node.add_child(left_cond, left)
            node.add_child(right_cond, right)
    
    return node_map[0] # root node

