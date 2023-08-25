import numpy as np

def save_data(test_x, test_y, fname='data'):
    data_dict = {'test_x': test_x, 'test_y': test_y}
    np.savez(f'{fname}.npz', **data_dict)

def load_data(fname='data'):
    loaded = np.load(f'{fname}.npz')
    return loaded['test_x'], loaded['test_y']

X_test, y_test = load_data('data_higgs')

# Shuffle each column independently
for i in range(X_test.shape[1]):
    X_test[:, i] = np.random.permutation(X_test[:, i])

y_test = np.random.permutation(y_test)

save_data(X_test, y_test, 'data_higgs_ood')