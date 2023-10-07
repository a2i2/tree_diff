import random
import sklearn.tree
import pandas as pd
from tree_diff import tree, keep_regrow_alg
from sklearn.tree import export_text
from sklearn.metrics import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from river.datasets import synth
from river import evaluate
from river import metrics
from river import tree as river_tree
from river import stream
import matplotlib.pyplot as plt
import time
import sys
import traceback

from tree_diff.tree_ruleset_conversion import *
from tree_diff.similar_tree import * 
from tree_diff.conversion import * 
import tree_diff.tree_metrics as tree_metrics
from tree_diff import tree, keep_regrow_alg

DATA_DIR = "../datasets"
OUT_DIR = "out16"

# Create subsequent batches of dataset  
def create_batches(X, y, n=2, max_batch_size=float('inf'), max_test_size=float('inf')):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=2)
    
    test_size = min(len(y_test), max_test_size)
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]

    num_batches = n
    batch_size = min(len(X_train) // num_batches, max_batch_size)

    # Data is already shuffled (so no need to shuffle again)
    # # Shuffle the dataframe
    # df = df.sample(frac=1).reset_index(drop=True)

    #Divide the dataframe into batches
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = (X_train.iloc[start:end], y_train.iloc[start:end])
        batches.append(batch)

    return batches, X_test, y_test


def compute_performance(model_names, batches, features, X_test, y_test, datasetname='dataset'):
    accuracy = []

    if 'efdt' in model_names:
        try:
            accuracy += eval_efdt(batches, features, X_test, y_test, datasetname)
        except Exception as e:
            print(f"Caught exception {e} in efdt")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)


    if 'keep-regrow' in model_names:
        try:
            accuracy += eval_keep_regrow(batches, features, X_test, y_test, datasetname)
        except Exception as e:
            print(f"Caught exception {e} in keep-regrow")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)            

    if 'tree-retrain' in model_names:
        try:
            accuracy += eval_tree_retrain(batches, features, X_test, y_test, datasetname)
        except Exception as e:
            print(f"Caught exception {e} in tree-retrain")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)

    return pd.DataFrame(accuracy)


def eval_keep_regrow(batches, features, X_test, y_test, datasetname):
    accuracy = []
    batch_number = 1

    X_batch_train, y_batch_train = batches[0]
    X_batch_test, y_batch_test = X_test, y_test
    
    print("Start of train block.")
    start_time = time.time()  # Record the start time
    # TODO: Infinite depth
    batch_tree = keep_regrow_alg.grow_tree(
        pd.DataFrame(X_batch_train, columns=features),
        y_batch_train,
        alpha = 10,
        beta = 0,
        grow_func = keep_regrow_alg.sklearn_grow_func,
        max_depth = float('inf')
    )
    end_time = time.time()    # Record the end time
    print("End of train block.")
    train_duration = end_time - start_time  # Calculate the duration


    X_batch_test_np = X_batch_test.to_numpy()
    y_batch_test_np = y_batch_test.to_numpy()
    
    print("Start of test block.")
    start_time = time.time()  # Record the start time

    batch_y_pred = [batch_tree.predict(X_batch_test_np[i]) for i in range(0, y_batch_test_np.shape[0])]
    
    end_time = time.time()    # Record the end time
    print("End of test block.")
    test_duration = end_time - start_time  # Calculate the duration

    
    batch_accuracy = np.mean(batch_y_pred == y_batch_test_np)
    tree_metrics.save_tree(batch_tree, f'{OUT_DIR}/keep-regrow_batch_1_{datasetname}')
    nodes = tree_metrics.nodes(batch_tree)
    accuracy.append({'alg': 'keep-regrow', 'batch': batch_number, 'acc': batch_accuracy, 'nodes': nodes, 'dataset': datasetname, 'similarity': float('nan'),
                     'train-duration': train_duration, 'test-duration': test_duration})
    
    for (X_batch_two_train, y_batch_two_train) in batches[1:]:
        batch_number += 1

        X_batch_train = pd.concat([X_batch_train, X_batch_two_train], axis=0)
        y_batch_train = pd.concat([y_batch_train, y_batch_two_train], axis=0)
        
        print("Start of train block.")
        start_time = time.time()  # Record the start time
        # TODO: Infinite depth
        full_clf = keep_regrow_alg.grow_tree(
            pd.DataFrame(X_batch_train, columns=features),
            y_batch_train,
            old_tree = batch_tree,
            alpha = 10,
            beta = 1,
            grow_func = keep_regrow_alg.sklearn_grow_func,
            max_depth = float('inf')
        )
        end_time = time.time()    # Record the end time
        print("End of train block.")
        train_duration = end_time - start_time  # Calculate the duration

        similarity = rule_set_similarity(tuple_tree_conversion(full_clf), tuple_tree_conversion(batch_tree))

        X_batch_test_np = X_batch_test.to_numpy()
        y_batch_test_np = y_batch_test.to_numpy()

        print("Start of test block.")
        start_time = time.time()  # Record the start time
        batch_y_pred = [full_clf.predict(X_batch_test_np[i]) for i in range(0, y_batch_test_np.shape[0])]
        end_time = time.time()    # Record the end time
        print("End of test block.")
        test_duration = end_time - start_time  # Calculate the duration

        batch_accuracy = np.mean(batch_y_pred == y_batch_test_np)
        
        nodes = tree_metrics.nodes(full_clf)
        tree_metrics.save_tree(full_clf, f'{OUT_DIR}/keep-regrow_batch_{batch_number}_{datasetname}')
        accuracy.append({'alg': 'keep-regrow', 'batch': batch_number, 'acc': batch_accuracy, 'nodes': nodes, 'dataset': datasetname, 'similarity': similarity,
                         'train-duration': train_duration, 'test-duration': test_duration})
        
        # next batch (current tree becomes the previous tree)
        batch_tree = full_clf

    return accuracy


def eval_tree_retrain(batches, features, X_test, y_test, datasetname):
    accuracy = []
    batch_number = 1

    X_batch_train, y_batch_train = batches[0]
    X_batch_test, y_batch_test = X_test, y_test

    print("Start of train block.")
    start_time = time.time()  # Record the start time
    batch_tree = keep_regrow_alg.grow_tree(
        pd.DataFrame(X_batch_train, columns=features),
        y_batch_train,
        alpha = 10,
        beta = 0,
        grow_func = keep_regrow_alg.sklearn_grow_func,
        max_depth = float('inf')
    )
    end_time = time.time()    # Record the end time
    print("End of train block.")
    train_duration = end_time - start_time  # Calculate the duration

    X_batch_test_np = X_batch_test.to_numpy()
    y_batch_test_np = y_batch_test.to_numpy()

    print("Start of test block.")
    start_time = time.time()  # Record the start time
    batch_y_pred = [batch_tree.predict(X_batch_test_np[i]) for i in range(0, y_batch_test_np.shape[0])]
    end_time = time.time()    # Record the end time
    print("End of test block.")
    test_duration = end_time - start_time  # Calculate the duration

    batch_accuracy = np.mean(batch_y_pred == y_batch_test_np)
    tree_metrics.save_tree(batch_tree, f'{OUT_DIR}/tree-retrain_batch_1_{datasetname}')
    nodes = tree_metrics.nodes(batch_tree)
    accuracy.append({'alg': 'tree-retrain', 'batch': batch_number, 'acc': batch_accuracy, 'nodes': nodes, 'dataset': datasetname, 'similarity': float('nan'),
                     'train-duration': train_duration, 'test-duration': test_duration})

    for (X_batch_two_train, y_batch_two_train) in batches[1:]:
        batch_number += 1

        X_batch_train = pd.concat([X_batch_train, X_batch_two_train], axis=0)
        y_batch_train = pd.concat([y_batch_train, y_batch_two_train], axis=0)
        
        try:
            print("Start of train block.")
            start_time = time.time()  # Record the start time
            full_clf = keep_regrow_alg.grow_tree(
                pd.DataFrame(X_batch_train, columns=features),
                y_batch_train,
                alpha = 10,
                beta = 0,
                grow_func = keep_regrow_alg.sklearn_grow_func,
                max_depth = float('inf')
            )
            end_time = time.time()    # Record the end time
            print("End of train block.")
            train_duration = end_time - start_time  # Calculate the duration
        except:
            # Catch bug in EFDT. Todo file bug report
            print(f"Warn: caught {e} when training EFDT on {datasetname} batch {batch_number}")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            # just log results for iterations so far
            return accuracy


        similarity = rule_set_similarity(tuple_tree_conversion(full_clf), tuple_tree_conversion(batch_tree))

        X_batch_test_np = X_batch_test.to_numpy()
        y_batch_test_np = y_batch_test.to_numpy()
        
        print("Start of test block.")
        start_time = time.time()  # Record the start time
        batch_y_pred = [full_clf.predict(X_batch_test_np[i]) for i in range(0, y_batch_test_np.shape[0])]
        end_time = time.time()    # Record the end time
        print("End of test block.")
        test_duration = end_time - start_time  # Calculate the duration

        batch_accuracy = np.mean(batch_y_pred == y_batch_test_np)
        
        nodes = tree_metrics.nodes(full_clf)
        tree_metrics.save_tree(full_clf, f'{OUT_DIR}/tree-retrain_batch_{batch_number}_{datasetname}')
        accuracy.append({'alg': 'tree-retrain', 'batch': batch_number, 'acc': batch_accuracy, 'nodes': nodes, 'dataset': datasetname, 'similarity': similarity,
                         'train-duration': train_duration, 'test-duration': test_duration})
        
        # next batch (current tree becomes the previous tree)
        batch_tree = full_clf

    return accuracy


def eval_efdt(batches, features, X_test, y_test, datasetname):
    accuracy = []
    batch_number = 1

    X_batch_train, y_batch_train = batches[0]
    X_batch_test, y_batch_test = X_test, y_test

    model_batch = river_tree.ExtremelyFastDecisionTreeClassifier(
        leaf_prediction = 'mc'
    )
   
    metric = metrics.Accuracy()

    print("Start of train block.")
    start_time = time.time()  # Record the start time
    evaluate.progressive_val_score(stream.iter_pandas(X_batch_train, y_batch_train), model_batch, metric)
    end_time = time.time()    # Record the end time
    print("End of train block.")
    train_duration = end_time - start_time  # Calculate the duration

    print("Start of test block.")
    start_time = time.time()  # Record the start time
    y_start_pred = []
    for x,y in stream.iter_pandas(X_batch_test,y_batch_test):
        y_start_pred.append(model_batch.predict_one(x))
    end_time = time.time()    # Record the end time
    print("End of test block.")
    test_duration = end_time - start_time  # Calculate the duration

    batch_accuracy = np.mean(y_start_pred == y_batch_test)
    batch1_rules = Ruleset(river_extract_rules(model_batch._root,river_children, river_is_leaf))

    #pd.DataFrame({"y_start_pred":y_start_pred, "y_batch_test":y_batch_test}).to_csv("eval1.csv")
    tree_metrics.river_save_tree(model_batch, f'{OUT_DIR}/efdt_batch_{batch_number}_{datasetname}')
    nodes = tree_metrics.river_nodes(model_batch)
    accuracy.append({'alg': 'efdt', 'batch': batch_number, 'acc': batch_accuracy,
                     'nodes': nodes, 'dataset': datasetname, 'similarity': float('nan'),
                     'train-duration': train_duration, 'test-duration': test_duration})
    
    for (X_batch_two_train, y_batch_two_train) in batches[1:]:
        batch_number += 1

        print("Start of train block.")
        start_time = time.time()  # Record the start time
        # TODO: Test that this is updating the model with additional data rather than starting from scratch
        evaluate.progressive_val_score(stream.iter_pandas(X_batch_two_train, y_batch_two_train), model_batch, metric)
        end_time = time.time()    # Record the end time
        train_duration = end_time - start_time  # Calculate the duration
        print("End of train block.")
        
        print("Start of test block.")
        start_time = time.time()  # Record the start time
        y_start_pred = []
        for x,y in stream.iter_pandas(X_batch_test, y_batch_test):
            p = model_batch.predict_one(x)
            y_start_pred.append(p)
            # print("===")
            # print(p)
            # print(model_batch.predict_proba_one(x))
            # print(model_batch.debug_one(x))
        end_time = time.time()    # Record the end time
        print("End of test block.")
        test_duration = end_time - start_time  # Calculate the duration

        batch_accuracy = np.mean(y_start_pred == y_batch_test)
        batch2_rules = Ruleset(river_extract_rules(model_batch._root, river_children, river_is_leaf))

        tree_metrics.river_save_tree(model_batch, f'{OUT_DIR}/efdt_batch_{batch_number}_{datasetname}')
        nodes = tree_metrics.river_nodes(model_batch)

        try:
            similarity = rule_set_similarity(batch1_rules, batch2_rules)
        except ZeroDivisionError as e:
            # This can be the case if both rulesets have 0 length
            print(f"Warn: caught {e}")
            similarity = float('nan')
        
        #import pdb; pdb.set_trace()
        accuracy.append({'alg': 'efdt', 'batch': batch_number, 'acc': batch_accuracy,
                         'nodes': nodes, 'dataset': datasetname, 'similarity': similarity,
                         'train-duration': train_duration, 'test-duration': test_duration})
        
        #pd.DataFrame({"y_start_pred":y_start_pred, "y_batch_test":y_batch_two_test}).to_csv("eval2.csv")

        # next batch (current tree becomes the previous tree)
        batch1_rules = batch2_rules

    return accuracy


def process(datapath, label, columns=False, sep=',', max_batch_size=float('inf'), max_test_size=float('inf'), datasetname='dataset'):
    #runs = 30
    #runs = 4
    runs = 2
    batches_per_run = 8
    #batches_per_run = 2
    print(f"=== Processing {datasetname} ===")

    if columns:
        # no headers, need to set manually
        df = pd.read_csv(datapath, names=columns, sep=sep)
    else:
        # assume first row contains columns
        df = pd.read_csv(datapath, sep=sep)
    #df = df.sample(frac = 0.0001) # 1100 rows out of 11 million
    features = [l for l in list(df.columns) if not l == label] 
    batches, X_test, y_test = create_batches(df[features], df[label], batches_per_run * runs, max_batch_size, max_test_size)
    
    #model_names = ['efdt']
    model_names = ['efdt', 'keep-regrow', 'tree-retrain']
    #model_names = ['efdt', 'keep-regrow']
    #model_names = ['keep-regrow']
    
    print(f"X_test, {X_test.shape}, {X_test}")
    print(f"y_test, {y_test.shape}, {y_test}")
    for i, b in enumerate(batches):
        print(f"batch {i}, {len(b[0]), len(b[1])}")

    for r in range(runs):
        print(f"==== Run {r} ====")

        run_batches = batches[r * batches_per_run : (r + 1) * batches_per_run]
        accuracy_df = compute_performance(model_names, run_batches, features, X_test, y_test, f"{datasetname}_run_{r}")
        accuracy_df.to_csv(f"{OUT_DIR}/accuracy_{datasetname}_run_{r}.csv")

if __name__ == "__main__":
    BATCH_SIZE = 1000
    TEST_SIZE = 100000
    #TEST_SIZE = 10000

    # need to shuffle
    process(
        f"{DATA_DIR}/skin+segmentation/Skin_NonSkin.txt",
        "CLASS",
        ["B","G","R","CLASS"],
        '\t',
        BATCH_SIZE,
        TEST_SIZE,
        "Skin")
    # with 2,000 points tree gets left as-is by keep-regrow
    #process("../datasets/skin+segmentation/Skin_NonSkin.txt", "CLASS", ["B","G","R","CLASS"], '\t', 2000)

    #exit()

    process(
        f"{DATA_DIR}/higgs/HIGGS.csv",
        "prediction",
        ["prediction","lepton_pT","lepton_eta","lepton_phi","missing_energy_magnitude","missing_energy_phi","jet_1_pt","jet_1_eta","jet_1_phi","jet_1_b-tag","jet_2_pt","jet_2_eta","jet_2_phi","jet_2_b-tag","jet_3_pt","jet_3_eta","jet_3_phi","jet_3_b-tag","jet_4_pt","jet_4_eta","jet_4_phi","jet_4_b-tag","m_jj","m_jjj","m_lv","m_jlv","m_bb","m_wbb","m_wwbb"],
        ',',
        BATCH_SIZE,
        TEST_SIZE,
        "Higgs"
    )
    
    process(
        f"{DATA_DIR}/susy/SUSY.csv",
        "prediction",
        ["prediction", "lepton_1_pT", "lepton_1_eta", "lepton_1_phi", "lepton_2_pT", "lepton_2_eta", "lepton_2_phi", "missing_energy_magnitude", "missing_energy_phi", "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2", "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1"],
        ',',
        BATCH_SIZE,
        TEST_SIZE,
        "Susy"
    )

    # includes headers
    process(
        f"{DATA_DIR}/hepmass/all_train.csv",
        "# label",
        False,
        ',',
        BATCH_SIZE,
        TEST_SIZE,
        "Hepmass"
    )

    # need to handle multiclass (class is 0 to 9)
    process(
        f"{DATA_DIR}/poker+hand/poker-hand-training-true.data",
        "CLASS",
        ["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "CLASS"],
        ',',
        BATCH_SIZE,
        TEST_SIZE,
        "Poker"
    )
    
    # need to handle multiclass (Cover_Type is 1 to 7)
    covtype_cols = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
                    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"] + \
                   [f"Wilderness_Area_{i}" for i in range(4)] + \
                   [f"Soil_Type_{i}" for i in range(40)] + \
                   ["Cover_Type"]
    process(
        f"{DATA_DIR}/covertype/covtype.data",
        "Cover_Type",
        covtype_cols,
        ',',
        BATCH_SIZE,
        TEST_SIZE,
        "Cover"
    )
