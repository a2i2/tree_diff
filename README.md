# tree_diff

To get started:
* Install poetry (https://python-poetry.org/)
* `poetry install`
* `export PYTHONPATH="$PWD"`
* `poetry run jupyter notebook`
* Navigate to `tree_diff/notebooks` in the browser
* Done!

# Run pipeline
* `poetry run python -m tree_diff assembler=baseline mode=train input_path=<project directory>/input`

# Additional dependencies

* `dot` command line tool on path (brew install graphviz / conda install graphviz)
* `rulecosi` algorithm needs to be git pulled into a `3rdparty` directory:
  * Run `mkdir 3rdparty` from the project directory
  * `cd 3rdparty`
  * `git clone https://github.com/jobregon1212/rulecosi.git`

In order to run the experiments in the `tree-dff/notebooks/Evaluation.ipynb`. 

Install the dependencies using the command below: 

```console
$ pip install -r requirements.txt
```

# Datasets
The following datasets are used in the experiments and evaluation for our model with other decision trees:

* [`Adult`](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
* [`Mushroom`](https://www.kaggle.com/datasets/uciml/mushroom-classification)
* [`HIGGS`](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)
* [`Android Malware`](https://archive.ics.uci.edu/ml/machine-learning-databases/00622/TUANDROMD.csv)

After downloading these datasets, place them in the `tree-diff/notebooks` folder to run the experiments.