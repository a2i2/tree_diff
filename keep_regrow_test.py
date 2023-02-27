from sklearn.datasets import load_iris
import random
import sklearn.tree
import pandas as pd
from tree_diff import keep_regrow_alg


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    # Shuffle dataset
    random.seed(2)
    idx = random.sample(list(range(0,len(X))), k=len(X))
    X = X[idx]
    y = y[idx]

    # batch 1
    X1, y1 = X[:len(X) // 10], y[:len(X) // 10]
    # batch 1 + 2 (full dataset)
    X2, y2 = X, y
    
    print("Tree at t = 1:")
    tree1 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X1, columns=iris.feature_names),
        y1
    )
    print(tree1.pretty_print())

    print("\nTree at t = 2:")
    tree2 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X2, columns=iris.feature_names),
        y2,
        tree1
    )
    print(tree2.pretty_print())

    print("\nTree at t = 2 if we discarded old tree:")    
    tree3 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X2, columns=iris.feature_names),
        y2
    )
    print(tree3.pretty_print())


if __name__ == "__main__":
    main()

