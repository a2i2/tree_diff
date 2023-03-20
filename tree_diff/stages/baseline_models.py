import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from surround import Estimator

from ..config import Config
from .assembler_state import AssemblerState

from sklearn.ensemble import GradientBoostingClassifier

from ..rule_entities import ModelType, Ruleset, Rule, Operator, Condition
from ..training_utils import train_model

from pathlib import Path
import os
import sys

# Load RuleCosi+
rule_cosi = str(Path(Path(os.getcwd()), "3rdparty/rulecosi"))
if not rule_cosi in sys.path:
    sys.path.insert(0, rule_cosi)

import rulecosi

def import_operator(string):
    return {'eq': Operator.EQ,
            'gt': Operator.GT,
            'lt': Operator.LT,
            'ge': Operator.GE,
            'le': Operator.LE,
            'ne': Operator.NE}[string]

def rulecosi_rule_to_conditions(rule):
    conditions = frozenset([Condition(a[1].att_name,
                                      import_operator(a[1].op.__name__), a[1].value)
                            for a in rule.A])
    return Rule(label = rule.y, conditions=conditions)

def rulecosi_rules_to_converted_rules(results, key):
    rulecosi_rules = results['estimator'][-1][key].simplified_ruleset_.rules
    converted_rules = frozenset(map(rulecosi_rule_to_conditions, rulecosi_rules))
    return Ruleset(rules = converted_rules)


def oneHot(string_columns):
    return [("OneHotEncoder", ColumnTransformer([
        ("string", OneHotEncoder(handle_unknown='ignore'), string_columns)]))]


def oneHotToDense(string_columns):
    return [("OneHotEncoder", ColumnTransformer([
       ("string", OneHotEncoder(handle_unknown='ignore'), string_columns)])),
            ("ToDense", FunctionTransformer(lambda x: x.toarray(), accept_sparse=True))]


ALGORITHMS = [
    (rulecosi.RuleCOSIClassifier, dict(base_ensemble=GradientBoostingClassifier()), oneHotToDense, ModelType.RULE),
    (GradientBoostingClassifier, dict(), oneHotToDense, ModelType.TREE_ENSEMBLE),
    (DecisionTreeClassifier, dict(), oneHot, ModelType.TREE)
]

LOGGER = logging.getLogger(__name__)

class BaselineModels(Estimator):
    def estimate(self, state, config):
        pass

    def fit(self, state: AssemblerState, config: Config):
        for model, model_config, preprocessor, model_type in ALGORITHMS:
            clf = model(**model_config)
            LOGGER.info(f"Training model: {clf}")
            model_name="rullecosi"
            train_model(clf,
                        state.training_records,
                        state.labels,
                        preprocessor(state.string_columns),
                        config,
                        model_name)
