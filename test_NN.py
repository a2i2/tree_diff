from sklearn.metrics import accuracy_score
from Merge_NN import *

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
bounded_test_x = bounded_test_x[:5000]
bounded_test_y = bounded_test_y[:5000]

# need to include final layer (softmax) as the weights between the relu layer and the softmax affect output
# todo: include softmax layer when computing output, but not as part of activation region/pattern
# pred_y6 = classify_on_fly_merged(
#     coefs1, intercepts1, 3, coefs2, intercepts2, 3,
#     volume_weighted_output, bounded_test_x)

#point = np.array([-0.35042147, -0.93489531, -0.36387789,  0.81477701, -0.78641961, -0.12329875])
point = np.array([-0.4, -0.9, -0.3,  0.8, -0.7, -0.1])
decision_space1 = find_region(coefs1, intercepts1, 3, point)
print(decision_space1.boundaries[0].vertices)

# point = np.array([-0.35042147, -0.93489531, -0.36387789,  0.81477701, -0.78641961, -0.12329875])
# #point = np.array([-0.4, -0.9, -0.3,  0.8, -0.7, -0.1])
# decision_space1 = find_region(coefs1, intercepts1, 3, point)
# decision_space1



# for single class:
# pred_y6 = np.where(pred_y6[:,0] > 0, 1, 0)
# best_bounded_test_y = bounded_test_y
# accuracy_score(best_bounded_test_y, pred_y6)



