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
from river import tree
from river import stream
import matplotlib.pyplot as plt

from tree_diff.tree_ruleset_conversion import *
from tree_diff.similar_tree import * 
from tree_diff.conversion import * 


# Create subsequent batches of dataset  
def create_batches(X, y, n=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    num_batches = n
    batch_size = len(X_train) // num_batches

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


def compute_performance(model_names, batches, features, X_test, y_test):
    X_batch_train, y_batch_train = batches[0]
    X_batch_two_train, y_batch_two_train = batches[1]

    X_batch_two_train_full = pd.concat([X_batch_train, X_batch_two_train], axis=0)
    y_batch_two_train_full = pd.concat([y_batch_train, y_batch_two_train], axis=0)

    # use the same testing data to evaluate both batches
    X_batch_test, y_batch_test = X_test, y_test
    X_batch_two_test, y_batch_two_test = X_test, y_test
    
    similarity = []
    accuracy = []
    
    if 'keep-regrow' in model_names:
        from tree_diff import tree, keep_regrow_alg
        
        # TODO: Infinite depth
        batch_tree = keep_regrow_alg.grow_tree(
            pd.DataFrame(X_batch_train, columns=features),
            y_batch_train,
            alpha = 1,
            beta = 0,
            grow_func = keep_regrow_alg.sklearn_grow_func
        )
            
        full_clf = keep_regrow_alg.grow_tree(
            pd.DataFrame(X_batch_two_train_full, columns=features),
            y_batch_two_train_full,
            old_tree = batch_tree,
            alpha = 1,
            beta = 1,
            regrow_func = keep_regrow_alg.sklearn_grow_func
        )
        
        similarity.append({'alg': 'keep-regrow', 'similarity': rule_set_similarity(tuple_tree_conversion(full_clf), tuple_tree_conversion(batch_tree))})
    
        X_batch_test_np = X_batch_test.to_numpy()
        y_batch_test_np = y_batch_test.to_numpy()
        X_batch_two_test_np = X_batch_two_test.to_numpy()
        y_batch_two_test_np = y_batch_two_test.to_numpy()
        
        batch_y_pred = [batch_tree.predict(X_batch_test_np[i]) for i in range(0, y_batch_test_np.shape[0])]
        batch_accuracy = np.mean(batch_y_pred == y_batch_test_np)
        
        full_y_pred = [full_clf.predict(X_batch_two_test_np[i]) for i in range(0, X_batch_two_test_np.shape[0])]
        full_accuracy = np.mean(full_y_pred == y_batch_two_test_np)
        
        accuracy.append({'alg': 'keep-regrow', 'batch_one': batch_accuracy, 'batch_two': full_accuracy})

    if 'tree-retrain' in model_names:
        from tree_diff import tree, keep_regrow_alg
        
        batch_tree = keep_regrow_alg.grow_tree(
            pd.DataFrame(X_batch_train, columns=features),
            y_batch_train,
            alpha = 1,
            beta = 0,
            regrow_func = keep_regrow_alg.sklearn_grow_func
        )
        
        full_clf = keep_regrow_alg.grow_tree(
            pd.DataFrame(X_batch_two_train_full, columns=features),
            y_batch_two_train_full,
            alpha = 1,
            beta = 0,
            regrow_func = keep_regrow_alg.sklearn_grow_func
        )

        similarity.append({'alg': 'tree-retrain', 'similarity': rule_set_similarity(tuple_tree_conversion(full_clf), tuple_tree_conversion(batch_tree))})

        X_batch_test_np = X_batch_test.to_numpy()
        y_batch_test_np = y_batch_test.to_numpy()
        X_batch_two_test_np = X_batch_two_test.to_numpy()
        y_batch_two_test_np = y_batch_two_test.to_numpy()
        
        batch_y_pred = [batch_tree.predict(X_batch_test_np[i]) for i in range(0, y_batch_test_np.shape[0])]
        batch_accuracy = np.mean(batch_y_pred == y_batch_test_np)
        
        full_y_pred = [full_clf.predict(X_batch_two_test_np[i]) for i in range(0, X_batch_two_test_np.shape[0])]
        full_accuracy = np.mean(full_y_pred == y_batch_two_test_np)
        
        accuracy.append({'alg': 'tree-retrain', 'batch_one': batch_accuracy, 'batch_two': full_accuracy})

    if 'efdt' in model_names:
        from river import tree

        model_batch = tree.ExtremelyFastDecisionTreeClassifier()
       
        metric = metrics.Accuracy()
        evaluate.progressive_val_score(stream.iter_pandas(X_batch_train, y_batch_train), model_batch, metric)
        
        y_start_pred = []
        for x,y in stream.iter_pandas(X_batch_test,y_batch_test):
            y_start_pred.append(model_batch.predict_one(x))
        batch_accuracy = np.mean(y_start_pred == y_batch_test)
        batch1_rules = Ruleset(river_extract_rules(model_batch._root,river_children, river_is_leaf))

        # TODO: Test that this is updating the model with additional data rather than starting from scratch
        evaluate.progressive_val_score(stream.iter_pandas(X_batch_two_train, y_batch_two_train), model_batch, metric)

        y_start_pred = []
        for x,y in stream.iter_pandas(X_batch_two_test,y_batch_two_test):
            y_start_pred.append(model_batch.predict_one(x))
        full_accuracy = np.mean(y_start_pred == y_batch_two_test)
        batch2_rules = Ruleset(river_extract_rules(model_batch._root,river_children, river_is_leaf))


        #import pdb; pdb.set_trace()
        accuracy.append({'alg': 'efdt', 'batch_one': batch_accuracy, 'batch_two': full_accuracy})
        
        try:
            similarity.append({'alg': 'efdt', 'similarity': rule_set_similarity(batch1_rules, batch2_rules)})
        except ZeroDivisionError as e:
            # This can be the case if both rulesets have 0 length
            print(f"Warn: caught {e}")
            similarity.append({'alg': 'efdt', 'similarity': float('nan')})

            
        

    return pd.DataFrame(accuracy), pd.DataFrame(similarity)


def process(datapath, label, columns):
    df = pd.read_csv(datapath, names=columns)
    #df = df.sample(frac = 0.0001) # 1100 rows out of 11 million
    features = [l for l in list(df.columns) if not l == "prediction"] 
    batches, X_test, y_test = create_batches(df[features],df["prediction"],2)
    model_names = ['keep-regrow', 'tree-retrain', 'efdt']
    #model_names = ['efdt']
    accuracy_df, similarity_df = compute_performance(model_names, batches, features, X_test, y_test)
    accuracy_df.to_csv("accuracy.csv")
    similarity_df.to_csv("similarity.csv")

if __name__ == "__main__":
    #process("HIGGS.csv", "prediction", ["prediction","lepton_pT","lepton_eta","lepton_phi","missing_energy_magnitude","missing_energy_phi","jet_1_pt","jet_1_eta","jet_1_phi","jet_1_b-tag","jet_2_pt","jet_2_eta","jet_2_phi","jet_2_b-tag","jet_3_pt","jet_3_eta","jet_3_phi","jet_3_b-tag","jet_4_pt","jet_4_eta","jet_4_phi","jet_4_b-tag","m_jj","m_jjj","m_lv","m_jlv","m_bb","m_wbb","m_wwbb"])
    #process("HIGGS_simp400.csv", "prediction", ["prediction","lepton_pT","lepton_eta","lepton_phi","missing_energy_magnitude","missing_energy_phi","jet_1_pt","jet_1_eta","jet_1_phi","jet_1_b-tag","jet_2_pt","jet_2_eta","jet_2_phi","jet_2_b-tag","jet_3_pt","jet_3_eta","jet_3_phi","jet_3_b-tag","jet_4_pt","jet_4_eta","jet_4_phi","jet_4_b-tag","m_jj","m_jjj","m_lv","m_jlv","m_bb","m_wbb","m_wwbb"])
    process("HIGGS_simp1000.csv", "prediction", ["prediction","lepton_pT","lepton_eta","lepton_phi","missing_energy_magnitude","missing_energy_phi","jet_1_pt","jet_1_eta","jet_1_phi","jet_1_b-tag","jet_2_pt","jet_2_eta","jet_2_phi","jet_2_b-tag","jet_3_pt","jet_3_eta","jet_3_phi","jet_3_b-tag","jet_4_pt","jet_4_eta","jet_4_phi","jet_4_b-tag","m_jj","m_jjj","m_lv","m_jlv","m_bb","m_wbb","m_wwbb"])
