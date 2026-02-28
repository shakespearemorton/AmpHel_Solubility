## Solubility Predictor — README

---

### 1. Overview
Two scripts:
- **`predict_solubility.py`** — CLI tool that takes a single peptide sequence and returns the probability of it being **soluble** (0 – 100 %)
- **`train_model.py`** — retrain the model whenever new peptides are added to `peptides.xlsx`

---

### 2. Installation
```bash
conda create -n solpred python=3.9 -y
conda activate solpred
pip install localcider pandas openpyxl joblib xgboost scikit-learn
```

---

### 3. Retraining the model
Add new peptides to the `for Training` sheet in `peptides.xlsx` (columns: `Name | Sequence | Experimental solubility`), then run:

```bash
python train_model.py
# → Saves williams_model.joblib
```

---

### 4. Inference on a single peptide
Sequences with Ac- / -NH2 termini, internal dashes, or repeat notation (e.g. `(K5)`) are cleaned automatically.

```bash
python predict_solubility.py "KKQFFFEYKLMLSMAKFESAM"
# → Predicted solubility: 95.7 %

python predict_solubility.py "Ac-GGGGKKK(K4)-NH2"
# → Predicted solubility: 78.3 %
```

---

### 5. Features
The model uses 5 biophysical descriptors computed via `localcider`:

| Feature | Description |
|---------|-------------|
| `fracpol` | Fraction of polar residues (Q, N, S, T, G, C, H) |
| `dispro` | Fraction of disorder-promoting residues |
| `isopoi` | Isoelectric point |
| `scd` | Sequence charge decoration (Sawle & Ghosh, JCP 2015) |
| `faro` | Fraction of aromatic residues (F, W, Y) |

Class imbalance (~2:1 soluble/insoluble) is handled by `scale_pos_weight` in XGBoost.

---

### 6. Performance
Hold-out test (65 sequences, 20% of 325):

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.754** |
| **MCC** | **0.412** |
| **ROC AUC** | **0.782** |

```
           Pred 0   Pred 1
Actual 0       11       10
Actual 1        6       38
```

---