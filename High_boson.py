import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def save_weights(weights_list, biases_list, fname='weights_and_biases'):
    weights_and_biases_dict = {f'weights_{i}': weights for i, weights in enumerate(weights_list)}
    weights_and_biases_dict.update({f'biases_{i}': biases for i, biases in enumerate(biases_list)})
    weights_and_biases_dict.update({'number_layers': len(weights_list)})

    np.savez(f'{fname}.npz', **weights_and_biases_dict)

def load_weights(fname='weights_and_biases'):
    loaded = np.load(f'{fname}.npz')
    number_layers = loaded['number_layers']
    weights_list = [loaded[f'weights_{i}'] for i in range(0, number_layers)]
    biases_list = [loaded[f'biases_{i}'] for i in range(0, number_layers)]
    return weights_list, biases_list

def save_data(test_x, test_y, fname='data'):
    data_dict = {'test_x': test_x, 'test_y': test_y}
    np.savez(f'{fname}.npz', **data_dict)

def load_data(fname='data'):
    loaded = np.load(f'{fname}.npz')
    return loaded['test_x'], loaded['test_y']

def rand_initialisation(X_train, X_test, y_train, y_test, fname):
    # Model Architecture
    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], activation='relu')) # Input layer
    model.add(Dense(16, activation='relu')) # Hidden layer
    model.add(Dense(1, activation='sigmoid')) # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test))

    weights_list1 = []
    biases_list1 = []

    # Iterate over the dense layers of the model
    for layer in model.layers:
        # save all weights (including for the sigmoid layer)
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            weights_list1.append(weights)
            biases_list1.append(biases)

    save_weights(weights_list1, biases_list1, f'weights_and_biases_{fname}')
    
    model2 = Sequential()
    model2.add(Dense(16, input_dim=X_train.shape[1], activation='relu')) # Input layer
    model2.add(Dense(16, activation='relu')) # Hidden layer
    model2.add(Dense(1, activation='sigmoid')) # Output layer
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test))

    weights_list2 = []
    biases_list2 = []

    # Iterate over the dense layers of the model
    for layer in model2.layers:
        # save all weights (including for the sigmoid layer)
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            weights_list2.append(weights)
            biases_list2.append(biases)

    save_weights(weights_list2, biases_list2, f'weights_and_biases_{fname}2')

    save_data(X_train, y_train, f'train_data_higgs_{fname}_1')
    save_data(X_train, y_train, f'train_data_higgs_{fname}_2')


def rand_subsets(X_train, X_test, y_train, y_test, fname, random_state = 32, split = 0.5):

    X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size= split, random_state=random_state)
    
    # Model Architecture
    model = Sequential()
    model.add(Dense(16, input_dim=X_train1.shape[1], activation='relu')) # Input layer
    model.add(Dense(16, activation='relu')) # Hidden layer
    model.add(Dense(1, activation='sigmoid')) # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train1, y_train1, epochs=4, batch_size=32, validation_data=(X_test, y_test))

    weights_list1 = []
    biases_list1 = []

    # Iterate over the dense layers of the model
    for layer in model.layers:
        # save all weights (including for the sigmoid layer)
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            weights_list1.append(weights)
            biases_list1.append(biases)

    save_weights(weights_list1, biases_list1, f'weights_and_biases_{fname}')
    
    model2 = Sequential()
    model2.add(Dense(16, input_dim=X_train2.shape[1], activation='relu')) # Input layer
    model2.add(Dense(16, activation='relu')) # Hidden layer
    model2.add(Dense(1, activation='sigmoid')) # Output layer
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model2.fit(X_train2, y_train2, epochs=4, batch_size=32, validation_data=(X_test, y_test))

    weights_list2 = []
    biases_list2 = []

    # Iterate over the dense layers of the model
    for layer in model2.layers:
        # save all weights (including for the sigmoid layer)
        if isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            weights_list2.append(weights)
            biases_list2.append(biases)

    save_weights(weights_list2, biases_list2, f'weights_and_biases_{fname}2')

    save_data(X_train1, y_train1, f'train_data_higgs_{fname}_1')
    save_data(X_train2, y_train2, f'train_data_higgs_{fname}_2')

columns = ["prediction","lepton_pT","lepton_eta","lepton_phi","missing_energy_magnitude","missing_energy_phi","jet_1_pt","jet_1_eta","jet_1_phi","jet_1_b-tag","jet_2_pt","jet_2_eta","jet_2_phi","jet_2_b-tag","jet_3_pt","jet_3_eta","jet_3_phi","jet_3_b-tag","jet_4_pt","jet_4_eta","jet_4_phi","jet_4_b-tag","m_jj","m_jjj","m_lv","m_jlv","m_bb","m_wbb","m_wwbb"]

higgs_boson_train = pd.read_csv("HIGGS.csv", names = columns)
#higgs_boson_train = pd.read_csv("notebooks/higgs-boson/training.csv")

# use all features (other than EventId and weight), we will apply PCA to reduce dimensionality
#higgs_boson_train_simple = higgs_boson_train.copy()
#higgs_boson_train_simple.drop(['EventId', 'Weight'], axis=1)
#df = higgs_boson_train_simple

# We use .03 percent of the data 
higgs_boson_simple = higgs_boson_train.sample(frac = .03)

df = higgs_boson_simple

# Preprocessing
encoder = LabelEncoder()
df['prediction'] = encoder.fit_transform(df['prediction'])

X = df.drop('prediction', axis=1)
y = df['prediction']


# normalise inputs to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cut down amount of training data so runs quickly
# X_train = X_train[:1000]
# y_train = y_train[:1000]

save_data(X_test, y_test, 'data_higgs')


for i in range(5): # do 5 different initialisations
    # Generate weights and biases file for a random initialisation
    rand_initialisation(X_train, X_test, y_train, y_test, f"randinit_higgs_{i}")

