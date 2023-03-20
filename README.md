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
  * `git pull https://github.com/jobregon1212/rulecosi.git`