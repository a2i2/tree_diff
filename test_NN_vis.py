from sklearn.metrics import accuracy_score
from Merge_NN import *
import cdd
import pandas as pd

# test_x, test_y = load_data('data_higgs')

# # drop points that are outside boundary
# bounded_test_x = test_x[
#     (test_x > -1).all(axis=1) & (test_x < 1).all(axis=1)
# ]
# bounded_test_y = test_y[
#     (test_x > -1).all(axis=1) & (test_x < 1).all(axis=1)
# ]

# just test on a subset of the data
# bounded_test_x = bounded_test_x[:5000]
# bounded_test_y = bounded_test_y[:5000]

def compare_dist_vis(fname):
    coefs1, intercepts1 = load_weights(f"weights_and_biases_{fname}")
    coefs2, intercepts2 = load_weights(f"weights_and_biases_{fname}2")

    X_train1, y_train1 = load_data(f"train_data_higgs_{fname}_1")
    X_train2, y_train2 = load_data(f"train_data_higgs_{fname}_2")

    # drop points that are outside boundary
    bounded_X_train1 = X_train1[
        (X_train1 > -1).all(axis=1) & (X_train1 < 1).all(axis=1)
    ]
    # bounded_y_train1 = y_train1[
    #     (X_train1 > -1).all(axis=1) & (X_train1 < 1).all(axis=1)
    #]
    bounded_X_train2 = X_train2[
        (X_train2 > -1).all(axis=1) & (X_train2 < 1).all(axis=1)
    ]
    # bounded_y_train2 = y_train2[
    #     (X_train2 > -1).all(axis=1) & (X_train2 < 1).all(axis=1)
    # ]

    # just test on a subset of the data
    # bounded_X_train1 = bounded_X_train1[:5]
    # bounded_X_train2 = bounded_X_train2[:5]

    boundaries1_to_points, boundaries1_to_volume = tabulate_points_in_regions(coefs1, intercepts1, 3, bounded_X_train1)
    boundaries2_to_points, boundaries2_to_volume = tabulate_points_in_regions(coefs2, intercepts2, 3, bounded_X_train2)

    # Create dataframes for each dictionary
    df_X = pd.DataFrame([(key, len(points), boundaries1_to_volume[key]) for key, points in boundaries1_to_points.items()], columns=['Key', 'Count', 'Volume'])
    df_Y = pd.DataFrame([(key, len(points), boundaries2_to_volume[key]) for key, points in boundaries2_to_points.items()], columns=['Key', 'Count', 'Volume'])

    # Save dataframes to CSV files
    df_X.to_csv(f'boundariesX_counts_{fname}.csv', index=False)
    df_Y.to_csv(f'boundariesY_counts_{fname}.csv', index=False)


for i in range(5):
    compare_dist_vis(f"randinit_higgs_{i}")

for i in range(5): # do 5 different random splits
    compare_dist_vis(f"subset_higgs_{i}")

for i in range(5): # do 5 different random splits (using 30/70 data split)
    compare_dist_vis(f"subset_0.7_higgs_{i}")

