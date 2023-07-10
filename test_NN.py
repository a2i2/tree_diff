from sklearn.metrics import accuracy_score
from Merge_NN import *
import cdd

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
decision_space1, empty_regions = find_region(coefs1, intercepts1, 3, point)

if len(decision_space1.boundaries) == 0:
    print(f"Empty region for point {point}")

print("empty regions:")
print(empty_regions)

assert len(empty_regions) == 1
H = empty_regions[0]

# TODO: confirm that cdd finds no solution
mat1 = cdd.Matrix(H, number_type = 'fraction')
mat1.rep_type = cdd.RepType.INEQUALITY
poly1 = cdd.Polyhedron(mat1)
#print("H matrix", poly1) # debug

gen = poly1.get_generators()
print("V matrix", gen) # debug

if gen.row_size == 0:
    # empty matrix, activation pattern does not exist
    print("No solution")

float_type = cdd.NumberTypeable('float')
matrix = np.matrix([[float_type.make_number(gen[i][j]) for j in range(0,gen.col_size)] for i in range(0,gen.row_size)])    

# check that cdd lib returned vertices (not rays)
if not np.all(matrix[:,0] == 1):
    #import pdb; pdb.set_trace()
    #assert False
    # for some reason there is a ray. Possibly a rounding error? Ignore (treat as no solution)
    print(f"Warning; ray found in matrix {matrix}")

# TODO: confirm that point is a solution
print("verticies:")
print(matrix)



#print(decision_space1.boundaries[0].vertices)

# point = np.array([-0.35042147, -0.93489531, -0.36387789,  0.81477701, -0.78641961, -0.12329875])
# #point = np.array([-0.4, -0.9, -0.3,  0.8, -0.7, -0.1])
# decision_space1 = find_region(coefs1, intercepts1, 3, point)
# decision_space1



# for single class:
# pred_y6 = np.where(pred_y6[:,0] > 0, 1, 0)
# best_bounded_test_y = bounded_test_y
# accuracy_score(best_bounded_test_y, pred_y6)



