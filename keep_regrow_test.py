from sklearn.datasets import load_iris
import random
import sklearn.tree
import pandas as pd
from tree_diff import keep_regrow_alg


def main():
    print(f"=== Regrow func: Sklearn regrowth algorithm ===\n")
    run_tests(keep_regrow_alg.sklearn_grow_func)

    print(f"=== Regrow func: Our tree regrowth algorithm ===\n")
    run_tests(keep_regrow_alg.tree_grow_func)
    
def run_tests(regrow_func):
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
    
    alpha = 1
    beta = 0
    
    print(f"Tree at t = 1 (alpha={alpha}, beta={beta}):")
    tree1 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X1, columns=iris.feature_names),
        y1,
        None,
        4,
        alpha,
        beta,
        regrow_func
    )
    print(tree1.pretty_print() + '\n')
    
    print(f"Tree at t = 2 (alpha={alpha}, beta={beta}):")
    tree2 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X2, columns=iris.feature_names),
        y2,
        tree1,
        4,
        alpha,
        beta,
        regrow_func
    )
    print(tree2.pretty_print() + '\n')

    alpha = 1
    beta = 0.25
    
    print(f"Tree at t = 2 (alpha={alpha}, beta={beta}):")
    tree2 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X2, columns=iris.feature_names),
        y2,
        tree1,
        4,
        alpha,
        beta,
        regrow_func
    )
    print(tree2.pretty_print() + '\n')
    
    alpha = 1
    beta = 1
    
    print(f"Tree at t = 2 (alpha={alpha}, beta={beta}):")
    tree2 = keep_regrow_alg.grow_tree(
        pd.DataFrame(X2, columns=iris.feature_names),
        y2,
        tree1,
        4,
        alpha,
        beta,
        regrow_func
    )
    print(tree2.pretty_print() + '\n')


if __name__ == "__main__":
    main()

