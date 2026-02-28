#!/usr/bin/env python3
"""
predict_solubility.py
---------------------
Compute biophysical descriptors for a single peptide and return the model’s
predicted probability of being **soluble** (0–100 %).

Usage
-----
$ python predict_solubility.py "Ac-GGGGKKK(N3)4-NH2"

Requirements
------------
localcider • pandas • joblib
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from localcider.sequenceParameters import SequenceParameters

import warnings
import xgboost
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
xgboost.set_config(verbosity=0)

# ───────────────────────────── paths ────────────────────────────────
MODEL_FILE = Path("williams_model.joblib")       # produced by ml_pipeline.py
RANDOM_STATE = 42                       # unused – kept for completeness

# ─────────────────── regexes for sequence hygiene ───────────────────
CLEAN_REGEXES = {
    "acetyl": re.compile(r"^\s*Ac-", flags=re.I),
    "amid":   re.compile(r"-NH2\s*$", flags=re.I),
    "repeat": re.compile(r"\(([ACDEFGHIKLMNPQRSTVWY])(\d+)\)", flags=re.I),
}
VALID_SEQ = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", flags=re.I)

# ───────────────────── preprocessing helpers ────────────────────────
def clean_sequence(raw: str) -> str:
    """Remove N-terminal Ac-, C-terminal –NH2, and expand (A4) style repeats."""
    seq = CLEAN_REGEXES["acetyl"].sub("", raw)
    seq = CLEAN_REGEXES["amid"].sub("", seq)
    seq = seq.replace("-", "")                         # strip internal dashes
    # expand e.g. (K5) → KKKKK
    while (m := CLEAN_REGEXES["repeat"].search(seq)):
        seq = seq[: m.start()] + m.group(1).upper() * int(m.group(2)) + seq[m.end():]
    seq = seq.strip().upper()
    if not VALID_SEQ.fullmatch(seq):
        sys.exit(f"[ERROR] Non-canonical residue(s) after cleaning: {seq}")
    return seq


def calc_sequence_features(seq: str) -> Dict[str, float]:
    """Return the exact 5 feature values used during model training."""
    sp    = SequenceParameters(seq)
    fracs = sp.get_amino_acid_fractions()
    return {
        "fracpol": sum(fracs[x] for x in "QNSTGCH"),
        "dispro":  sp.get_fraction_disorder_promoting(),
        "isopoi":  sp.get_isoelectric_point(),
        "scd":     sp.get_SCD(),
        "faro":    sum(fracs[x] for x in "FWY"),
    }

# ─────────────────────────── CLI glue ───────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Predict peptide solubility (probability 0–100 %)."
    )
    ap.add_argument("sequence", help="Raw peptide sequence (Ac-/-NH2 allowed).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ---- sanitise & featurise -------------------------------------------------
    seq = clean_sequence(args.sequence)
    feats = calc_sequence_features(seq)
    X = pd.DataFrame([feats])            # shape (1, 5)

    # ---- load model ----------------------------------------------------------
    if not MODEL_FILE.exists():
        sys.exit("[ERROR] model.joblib not found — train the model first.")
    payload = joblib.load(MODEL_FILE)
    model = payload["model"]             # pipeline (imputer → scaler → xgb)

    # ---- inference -----------------------------------------------------------
    prob_sol = model.predict_proba(X)[:, 1][0]          # class 1 = soluble
    print(f"Predicted solubility: {prob_sol*100:.1f} %")

if __name__ == "__main__":
    main()