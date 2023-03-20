import logging

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from .config import Config
from .rule_entities import Ruleset

LOGGER = logging.getLogger(__name__)
SCORING = {'roc_auc': 'roc_auc',
           'f1_macro': 'f1_macro',
           'balanced_accuracy': 'balanced_accuracy'
           }


def rule_overlap(ruleset: Ruleset):
    all_conditions = [i for r in ruleset for i in r.conditions]
    return len(set(all_conditions)) / len(all_conditions)


def rule_sparsity(ruleset: Ruleset):
    return sum([len(r) for r in ruleset]) / len(ruleset)


def report_metrics(scores, scoring):
    strings = []
    for k in scoring.keys():
        key = f"test_{k}"
        s = f"{scores[key].mean():0.2f}"
        strings.append(f"{k}: {s}")
    return ", ".join(strings)

def var_name(var):
    vars = [v for v in globals() if globals()[v] == var]
    return vars[0] if len(vars) > 0 else None

def train_model(model, X, y, preprocessor, config: Config, model_name=None):
    pipeline = []
    if preprocessor:
        for p in preprocessor:
            pipeline.append(p)
    pipeline.append((model_name if model_name else str(model), model))
    pipeline = Pipeline(pipeline)
    scores = cross_validate(pipeline, X, y, return_estimator=True, cv=config.cv, scoring=SCORING)
    LOGGER.info(f"{str(model)} : {report_metrics(scores, SCORING)}")
    return scores


