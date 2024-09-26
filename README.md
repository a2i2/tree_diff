# tree_diff

This repository contains code for the paper [Minimising changes to audit when updating decision trees](https://arxiv.org/abs/2408.16321).

# Interactive notebooks

We provide notebooks that can be used to explore how the algorithm works without running the full experiment.

To get started with the interactive notebooks:
* Install poetry (https://python-poetry.org/)
* `poetry install`
* `export PYTHONPATH="$PWD"`
* `poetry run jupyter notebook`
* Navigate to `tree_diff/notebooks` in the browser
* Done!

# Additional dependencies

We use the `dot` command line tool for generating tree figures:
* `dot` command line tool on path (brew install graphviz / conda install graphviz)

The full experiment was run in a conda environment. In addition to standard conda dependencies (sklearn, pandas, numpy, matplotlib), comparisons to EFDT need the river library. Install the additional dependencies using the command below: 

```console
$ pip install -r requirements.txt
```

# Datasets

Datasets for the paper are available from the UCI Machine Learning Repository at https://archive.ics.uci.edu. Extract the datasets to a directory called `datasets`. Your folder structure should look like this: 

```
tree_diff/
└...

datasets/
├── covertype
│   ├── covtype.data
│   ├── covtype.info
│   └── old_covtype.info
├── covertype.zip
├── hepmass
│   ├── 1000_test.csv
│   ├── 1000_train.csv
│   ├── all_test.csv
│   ├── all_train.csv
│   ├── not1000_test.csv
│   └── not1000_train.csv
├── hepmass.zip
├── higgs
│   └── HIGGS.csv
├── higgs.zip
├── poker+hand
│   ├── poker-hand.names
│   ├── poker-hand-testing.data
│   └── poker-hand-training-true.data
├── poker+hand.zip
├── skin+segmentation
│   └── Skin_NonSkin.txt
├── skin+segmentation.zip
├── susy
│   └── SUSY.csv
└── susy.zip
```

# Running the experiments

After downloading the datasets, you can run the experiment in the paper:

```
python3 experiment2.py
```

# Analysis of experiment results

The experiment will write results to an output directory (starting with `out` followed by a number). The notebook used to analyse these experiment results is `notebooks/experiment2_analysis.ipynb`

