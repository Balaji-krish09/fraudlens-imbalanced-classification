"""Shared helpers: data loading, pipelines, samplers, evaluation."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from .config import (
    RAW_DATA, PROCESSED_DIR, TARGET, SCALE_COLS, RANDOM_STATE, CV_FOLDS,
)


# ---------------------------------------------------------------- data
def load_raw() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"{RAW_DATA} not found. Download creditcard.csv from Kaggle "
            "(mlg-ulb/creditcardfraud) and put it in data/raw/."
        )
    return pd.read_csv(RAW_DATA)


def load_splits():
    """Train/test splits saved by 02_baseline.ipynb."""
    train = pd.read_pickle(PROCESSED_DIR / "train.pkl")
    test = pd.read_pickle(PROCESSED_DIR / "test.pkl")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- pipelines
def get_preprocessor() -> ColumnTransformer:
    """RobustScaler on Time/Amount (Amount has heavy outliers); V1-V28 pass through."""
    return ColumnTransformer(
        [("scale", RobustScaler(), SCALE_COLS)],
        remainder="passthrough",
    )


def make_sampler(name):
    """Resampling strategies. They live INSIDE the pipeline so that during CV
    they only ever touch the training folds, never the validation fold."""
    if name == "undersample":
        # 1:1 balance -> throws away ~99.8% of legit rows
        return RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE)
    if name == "smote":
        # Synthesize frauds until they are 10% of legit. Full 1:1 would create
        # ~180k synthetic points from ~390 real ones: slow and mostly noise.
        return SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
    return None


def build_pipeline(model, sampler_name=None) -> Pipeline:
    steps = [("prep", get_preprocessor())]
    sampler = make_sampler(sampler_name)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


# ---------------------------------------------------------------- evaluation
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
}


def get_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_evaluate(pipe, X, y, name=None) -> dict:
    """Stratified k-fold CV. Returns mean (and std of PR-AUC) for each metric."""
    res = cross_validate(pipe, X, y, cv=get_cv(), scoring=SCORING, n_jobs=1)
    out = {"name": name} if name else {}
    for k in SCORING:
        out[k] = res[f"test_{k}"].mean()
    out["pr_auc_std"] = res["test_pr_auc"].std()
    out["fit_time_s"] = res["fit_time"].mean()
    return out


def metrics_at_threshold(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ---------------------------------------------------------------- models
MODEL_NAMES = ["LogReg", "RandomForest", "XGBoost", "LightGBM"]
TECHNIQUES = ["None", "Class weights", "Undersampling", "SMOTE"]


def make_model(name, weighted=False, pos_weight=1.0):
    """Build a classifier. `weighted=True` turns on each library's own way of
    up-weighting the fraud class. `pos_weight` = n_legit / n_fraud (for XGB/LGBM)."""
    if name == "LogReg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                  class_weight="balanced" if weighted else None)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                      random_state=RANDOM_STATE,
                                      class_weight="balanced_subsample" if weighted else None)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE,
                             scale_pos_weight=pos_weight if weighted else 1.0)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
                              scale_pos_weight=pos_weight if weighted else 1.0)
    raise ValueError(f"Unknown model: {name}")


def make_experiment(model_name, technique, pos_weight=1.0) -> Pipeline:
    """model x technique -> full pipeline (preprocess -> [sampler] -> model)."""
    sampler = {"Undersampling": "undersample", "SMOTE": "smote"}.get(technique)
    model = make_model(model_name, weighted=(technique == "Class weights"), pos_weight=pos_weight)
    return build_pipeline(model, sampler_name=sampler)
