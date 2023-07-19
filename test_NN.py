from sklearn.metrics import accuracy_score
from Merge_NN import *
import cdd
import pandas as pd

coefs1, intercepts1 = load_weights("notebooks/weights_and_biases_higgs")
coefs2, intercepts2 = load_weights("notebooks/weights_and_biases_higgs2")
test_x, test_y = load_data('notebooks/data_higgs')

#decision_space_1_region = find_region(coefs1, intercepts1, 1, point)
#plot_decision_space_nn(decision_space_1_region,0,False)

# drop points that are outside boundary
bounded_test_x = test_x[
    (test_x > -1).all(axis=1) & (test_x < 1).all(axis=1)
]
bounded_test_y = test_y[
    (test_x > -1).all(axis=1) & (test_x < 1).all(axis=1)
]

# just test on a subset of the data
#bounded_test_x = bounded_test_x[:5000]
#bounded_test_y = bounded_test_y[:5000]
bounded_test_x = bounded_test_x[:20]
bounded_test_y = bounded_test_y[:20]

def try_stagegy(strategy):
    pred_y = classify_on_fly_merged(
    coefs1, intercepts1, 3, coefs2, intercepts2, 3,
    strategy, bounded_test_x)

    acc = nn_accuracy(pred_y, bounded_test_y)
    loss = nn_log_loss(pred_y, bounded_test_y)

    return acc, loss

acc_vol, loss_vol = try_stagegy(volume_weighted_output)
acc_global_avg, loss_global_avg = try_stagegy(avg_output)

data = {
    'Strategy': ['Volume', 'Average'],
    'Accuracy': [acc_vol, acc_global_avg],
    'Loss': [loss_vol, loss_global_avg],
}

df = pd.DataFrame(data)
df.to_csv("merge_results.csv")