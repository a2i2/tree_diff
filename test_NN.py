from sklearn.metrics import accuracy_score
from Merge_NN import *
import cdd
import pandas as pd
import sys

do_volume_computation = True
if len(sys.argv) > 3:
    arg = sys.argv[3]
    if arg.lower() == "skipvol":
        print("volume computation disabled")
        do_volume_computation = False
if len(sys.argv) > 2:
    test_file = sys.argv[2]
    test_x, test_y = load_data(f'{test_file}')
    cust_test_name = f"_{test_file}"
    print(f"Using custom test file: {test_file}")
else:
    test_x, test_y = load_data('data_higgs')
    cust_test_name = ""
    print("Using default test file")

#decision_space_1_region = find_region(coefs1, intercepts1, 1, point)
#plot_decision_space_nn(decision_space_1_region,0,False)

# drop points that are outside boundary
#import pdb; pdb.set_trace()
print("shape before filtering", test_x.shape)
bounded_test_x = test_x[
    (test_x > -1).all(axis=1) & (test_x < 1).all(axis=1)
]
bounded_test_y = test_y[
    (test_x > -1).all(axis=1) & (test_x < 1).all(axis=1)
]
print("shape after filtering", bounded_test_x.shape)

# DISABLE Check between -1 and 1 (temporary hack)
bounded_test_x = test_x
bounded_test_y = test_y

# just test on a subset of the data
# bounded_test_x = bounded_test_x[:10000]
# bounded_test_y = bounded_test_y[:10000]
# bounded_test_x = bounded_test_x[:200]
# bounded_test_y = bounded_test_y[:200]

def try_stagegy(strategy, coefs1, intercepts1, coefs2, intercepts2):
    pred_y = classify_on_fly_merged(
    coefs1, intercepts1, 3, coefs2, intercepts2, 3,
    strategy, bounded_test_x)

    acc = nn_accuracy(pred_y, bounded_test_y)
    loss = nn_log_loss(pred_y, bounded_test_y)

    return acc, loss

# def try_data_density(strategy, coefs1, intercepts1, coefs2, intercepts2):
#     pred_y = ...

#     acc = nn_accuracy(pred_y, bounded_test_y)
#     loss = nn_log_loss(pred_y, bounded_test_y)

#     return acc, loss

def try_unmerged(coefs, intercepts):
    pred_y = classify_on_fly(
    coefs, intercepts, 3, bounded_test_x)

    acc = nn_accuracy(pred_y, bounded_test_y)
    loss = nn_log_loss(pred_y, bounded_test_y)

    return acc, loss

def compare_merge(fname):
    coefs1, intercepts1 = load_weights(f"weights_and_biases_{fname}")
    coefs2, intercepts2 = load_weights(f"weights_and_biases_{fname}2")
    acc_vol, loss_vol = try_stagegy(volume_weighted_output, coefs1, intercepts1, coefs2, intercepts2)
    acc_global_avg, loss_global_avg = try_stagegy(avg_output, coefs1, intercepts1, coefs2, intercepts2)
    acc1, loss1 = try_unmerged(coefs1, intercepts1)
    acc2, loss2 = try_unmerged(coefs2, intercepts2)

    data = {
        'Strategy': ['Volume', 'Average', 'NN1', 'NN2'],
        'Accuracy': [acc_vol, acc_global_avg, acc1, acc2],
        'Loss': [loss_vol, loss_global_avg, loss1, loss2],
    }

    df = pd.DataFrame(data)
    df.to_csv(f"merge_results_{fname}{cust_test_name}.csv")


def compute_merge_table(fname):
    coefs1, intercepts1 = load_weights(f"weights_and_biases_{fname}")
    coefs2, intercepts2 = load_weights(f"weights_and_biases_{fname}2")

    X_train1, y_train1 = load_data(f"train_data_higgs_{fname}_1")
    X_train2, y_train2 = load_data(f"train_data_higgs_{fname}_2")

    # Precompute number of points for training data

    # drop points that are outside boundary
    bounded_X_train1 = X_train1[
        (X_train1 > -1).all(axis=1) & (X_train1 < 1).all(axis=1)
    ]
    bounded_X_train2 = X_train2[
        (X_train2 > -1).all(axis=1) & (X_train2 < 1).all(axis=1)
    ]

    # DISABLE Check between -1 and 1 (temporary hack)
    bounded_X_train1 = X_train1
    bounded_X_train2 = X_train2

    results_outputX, results_volumeX, results_num_training_pointsX = extract_merge_attributes(coefs1, intercepts1, 3, bounded_X_train1, bounded_test_x, do_volume_computation)
    results_outputY, results_volumeY, results_num_training_pointsY = extract_merge_attributes(coefs2, intercepts2, 3, bounded_X_train2, bounded_test_x, do_volume_computation)

    # Create a pandas DataFrame using the numpy arrays
    #import pdb; pdb.set_trace()
    data_frame = pd.DataFrame({
        'point': [str(p) for p in bounded_test_x],
        'A_volume': results_volumeX,
        'A_num_training_points': results_num_training_pointsX,
        'B_volume': results_volumeY,
        'B_num_training_points': results_num_training_pointsY,
        'A_pred_logit': np.array(results_outputX)[:,0],
        'B_pred_logit': np.array(results_outputY)[:,0],
        'actual': bounded_test_y
    })

    # Save the DataFrame to a CSV file
    csv_filename = f"merge_attributes_{fname}{cust_test_name}.csv"
    print(f"saving results to {csv_filename}")
    data_frame.to_csv(csv_filename, index=False)


# for i in range(5):
#     compute_merge_table(f"randinit_higgs_{i}")

# for i in range(5): # do 5 different random splits
#     compute_merge_table(f"subset_higgs_{i}")

# for i in range(5): # do 5 different random splits (using 30/70 data split)
#     compute_merge_table(f"subset_0.7_higgs_{i}")

# Check if there's at least one argument
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    print("No input file provided")
    sys.exit(1)

for i in range(5):
    compute_merge_table(f"{input_file}_{i}")
    #break # only for first input (TESTING)
