#!/usr/bin/env python3
"""
train_model.py
--------------
Train an XGBoost solubility classifier from a peptide Excel file and save
the fitted model to williams_model.joblib.

Usage
-----
    python train_model.py

The script reads peptides.xlsx from the same directory. It expects a sheet
named "for Training" with columns:
    Name  |  Sequence  |  Experimental solubility   (1 = soluble, 0 = insoluble)

Sequences with Ac- / -NH2 modifications or (A4)-style repeats are cleaned
automatically before featurisation.

Requirements
------------
    localcider  sparrow  pandas  openpyxl  joblib  xgboost  scikit-learn
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from localcider.sequenceParameters import SequenceParameters
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
xgb.set_config(verbosity=0)

# ── Constants ────────────────────────────────────────────────────────────────
EXCEL_FILE   = Path("peptides.xlsx")
SHEET_NAME   = "for Training"
MODEL_FILE   = Path("williams_model.joblib")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

PARAM_GRID = {
    "xgb__n_estimators":  [100, 200, 400],
    "xgb__max_depth":     [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.1],
    "xgb__subsample":     [0.8, 1.0],
}

# ── Sequence cleaning ─────────────────────────────────────────────────────────
_ACETYL = re.compile(r"^\s*Ac-", flags=re.I)
_AMID   = re.compile(r"-NH2\s*$", flags=re.I)
_REPEAT = re.compile(r"\(([ACDEFGHIKLMNPQRSTVWY])(\d+)\)", flags=re.I)
_VALID  = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", flags=re.I)

def clean_sequence(raw: str) -> str:
    """Strip Ac- / -NH2 termini, internal dashes, and expand (K5)-style repeats."""
    seq = _ACETYL.sub("", raw)
    seq = _AMID.sub("", seq)
    seq = seq.replace("-", "")
    while (m := _REPEAT.search(seq)):
        seq = seq[: m.start()] + m.group(1).upper() * int(m.group(2)) + seq[m.end():]
    seq = seq.strip().upper()
    if not _VALID.fullmatch(seq):
        raise ValueError(f"Non-canonical residue(s) after cleaning: '{seq}'")
    return seq

# ── Feature calculation ───────────────────────────────────────────────────────
def calc_features(seq: str) -> dict[str, float]:
    """Compute the 5 biophysical features used during model training."""
    sp    = SequenceParameters(seq)
    fracs = sp.get_amino_acid_fractions()
    return {
        "fracpol":          sum(fracs[aa] for aa in "QNSTGCH"),
        "dispro":           sp.get_fraction_disorder_promoting(),
        "isopoi":           sp.get_isoelectric_point(),
        "scd":              sp.get_SCD(),
        "faro":             sum(fracs[aa] for aa in "FWY"),
    }

# ── Data loading ─────────────────────────────────────────────────────────────
def load_excel(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the training sheet and return (feature matrix X, label vector y)."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME)

    # Columns by position: Name | Sequence | Experimental solubility
    seq_col, label_col = df.columns[1], df.columns[2]
    df = df[[seq_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)

    n_sol   = df[label_col].sum()
    n_insol = (df[label_col] == 0).sum()
    print(f"  Loaded {len(df)} peptides from '{SHEET_NAME}'")
    print(f"  Soluble: {n_sol}  |  Insoluble: {n_insol}  "
          f"(scale_pos_weight = {n_sol / n_insol:.2f})")

    print("  Calculating features …")
    features, labels = [], []
    for seq, label in zip(df[seq_col], df[label_col]):
        try:
            features.append(calc_features(clean_sequence(str(seq))))
            labels.append(label)
        except Exception as e:
            print(f"  [WARN] Skipping '{seq}': {e}")

    return pd.DataFrame(features), pd.Series(labels, name="solubility")

# ── Model pipeline ────────────────────────────────────────────────────────────
def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("xgb",     xgb.XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

# ── Training ──────────────────────────────────────────────────────────────────
def train() -> None:
    print(f"\nLoading data from {EXCEL_FILE} …")
    X, y = load_excel(EXCEL_FILE)

    # Compute class weight from the full dataset and carry it through
    scale_pos_weight = (y == 1).sum() / (y == 0).sum()

    # Stratified hold-out: preserves the ~2:1 ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Hyperparameter search on the training split
    print(f"\nRunning {CV_FOLDS}-fold grid search …")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        build_pipeline(scale_pos_weight), PARAM_GRID,
        cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  CV ROC AUC  : {gs.best_score_:.3f}")

    # Evaluate on the held-out set
    best = gs.best_estimator_
    y_prob = best.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n── Hold-out results ({len(y_test)} peptides) ────────────────────")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"  MCC      : {matthews_corrcoef(y_test, y_pred):.3f}")
    print(f"  ROC AUC  : {roc_auc_score(y_test, y_prob):.3f}")
    print(f"  Confusion matrix (rows = actual, cols = predicted):")
    print(f"             Pred 0   Pred 1")
    print(f"  Actual 0   {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  Actual 1   {cm[1,0]:>6}   {cm[1,1]:>6}")
    print("────────────────────────────────────────────────────────────")

    # Refit on ALL data with the best hyperparameters and save
    final_model = build_pipeline(scale_pos_weight)
    final_model.set_params(**gs.best_params_)
    final_model.fit(X, y)

    joblib.dump({"model": final_model}, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not EXCEL_FILE.exists():
        sys.exit(f"[ERROR] '{EXCEL_FILE}' not found. Place it in the same directory.")
    train()