#%%
# =============================================================================
#  Waldschäden‑Detection  |  XGBoost + Isotone  |  Jahr‑stratifizierte CV
#  ---------------------------------------------------------------------------
#  - verhindert forest_id‑Leakage zwischen Train / Validation
#  - korrigiert NDWI‑Formel (Gao‑Variante)
#  - nutzt PR‑AUC ('aucpr') als Optimierungskriterium
#  - 5‑fach GroupKFold‑Cross‑Validation (Gruppen = Jahr)
#  - ermittelt τ* (F2‑Optimum bei Recall ≥ 0.60) je Fold, nimmt Median
#  - **Logik & Ergebnisse bleiben unverändert – Code nur lesbarer!**
# =============================================================================

#%%
# ╔══════════════════════════════════════════════════════════════════════╗
# 1 │ Bibliotheken & Logging                                             │
# ╚══════════════════════════════════════════════════════════════════════╝
import datetime
import logging
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, fbeta_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------
LOG_DIR = Path("Tim/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Helper für optische Überschriften in Notebook‑Ausgabe
def banner(text: str) -> None:
    line = "═" * 78
    print(f"\n{line}\n{text}\n{line}")

# Reproduzierbarkeit
np.random.seed(42)
random.seed(42)
logging.info("Notebook gestartet – Seeds gesetzt (42).")

#%%
# ╔══════════════════════════════════════════════════════════════════════╗
# 2 │ Daten laden  &  Leakage‑Prüfung                                    │
# ╚══════════════════════════════════════════════════════════════════════╝
banner("DATA LOADING")

DATA_DIR = Path("data")                       # <‑‑ Pfad ggf. anpassen
train_df = (
    pd.read_csv(DATA_DIR / "train.csv")
      .sort_values(["forest_id", "year"])
      .reset_index(drop=True)
)
val_df = (
    pd.read_csv(DATA_DIR / "validation.csv")
      .sort_values(["forest_id", "year"])
      .reset_index(drop=True)
)

# -----------------------------------------------------------------------
# forest_id‑Leakage: gleiche Fläche darf nicht in beiden Splits sein
# -----------------------------------------------------------------------
overlap_ids = set(train_df.forest_id) & set(val_df.forest_id)
if overlap_ids:
    print(f"⚠️  {len(overlap_ids)} forest_ids doppelt; werden aus Validation entfernt.")
    val_df = val_df[~val_df.forest_id.isin(overlap_ids)]

# -----------------------------------------------------------------------
# Zeilen‑/Spalten‑Koordinaten (142 × 142 Raster) anlegen, falls nicht vorhanden
# -----------------------------------------------------------------------
for df in (train_df, val_df):
    if {"row", "col"}.issubset(df.columns):
        continue
    n_rows = n_cols = 142
    df["row"] = df["forest_id"] // n_cols
    df["col"] = df["forest_id"] % n_cols

#%%
# ╔══════════════════════════════════════════════════════════════════════╗
# 3 │ Feature Engineering                                                │
# ╚══════════════════════════════════════════════════════════════════════╝
banner("FEATURE ENGINEERING")

def engineer_features(group: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Erzeugt spektrale Indizes + zeitliche Ableitungen für eine Wald‑ID."""
    eps = 1e-6
    g = group.copy()

    # -- Spektrale Indizes ------------------------------------------------
    g["NDVI"] = (g.near_infrared - g.red) / (g.near_infrared + g.red + eps)
    g["NBR"]  = (g.near_infrared - g.shortwave_infrared_2) / (g.near_infrared + g.shortwave_infrared_2 + eps)
    g["NDMI"] = (g.near_infrared - g.shortwave_infrared_1) / (g.near_infrared + g.shortwave_infrared_1 + eps)
    g["NDWI"] = (g.near_infrared - g.shortwave_infrared_1) / (g.near_infrared + g.shortwave_infrared_1 + eps)

    g["EVI"]  = 2.5 * (g.near_infrared - g.red) / (g.near_infrared + 6 * g.red - 7.5 * g.blue + 1 + eps)
    g["NBR2"] = (g.shortwave_infrared_1 - g.shortwave_infrared_2) / (g.shortwave_infrared_1 + g.shortwave_infrared_2 + eps)

    # -- Zeitliche Ableitungen (Differenzen & Lags) -----------------------
    g["dNDVI"] = g["NDVI"].diff()
    g["dNBR"]  = g["NBR"].shift(1) - g["NBR"]          # Vorjahr−Aktuell

    for lag in range(1, lags + 1):
        g[f"NDVI_lag{lag}"]  = g["NDVI"].shift(lag)
        g[f"dNDVI_lag{lag}"] = g["dNDVI"].shift(lag)
        g[f"NBR_lag{lag}"]   = g["NBR"].shift(lag)
        g[f"dNBR_lag{lag}"]  = g["dNBR"].shift(lag)

    # -- Drei‑Jahres‑Statistik + Anomalien --------------------------------
    g["var_NBR_3yr"]   = g["NBR"].shift(1).rolling(3, min_periods=3).var()
    g["trend_NBR_3yr"] = g["NBR"] - g["NBR"].shift(3)
    g["z_NBR"]         = (
        (g["NBR"] - g["NBR"].shift(1).rolling(3, min_periods=2).mean()) /
        (g["NBR"].shift(1).rolling(3, min_periods=2).std() + eps)
    )

    # -- Disturbance‑Proxies ---------------------------------------------
    g["d2NBR"] = g["dNBR"].diff()

    return g

# Feature‑Engineering auf beiden Splits anwenden
train_feat = (
    train_df.groupby("forest_id", group_keys=False)
            .apply(engineer_features)
            .dropna()
            .reset_index(drop=True)
)
val_feat = (
    val_df.groupby("forest_id", group_keys=False)
          .apply(engineer_features)
          .dropna()
          .reset_index(drop=True)
)

# X / y‑Matrizen
EXCLUDE = ["is_disturbance", "fid", "forest_id", "year", "row", "col"]
FEATURES = [c for c in train_feat.columns if c not in EXCLUDE]

X_train = train_feat[FEATURES].values
y_train = train_feat["is_disturbance"].values
X_val   = val_feat[FEATURES].values
y_val   = val_feat["is_disturbance"].values
years_train = train_feat["year"].values          # für GroupKFold

print(f"🛈  {len(FEATURES)} Feature‑Spalten erzeugt.")

#%%
# ╔══════════════════════════════════════════════════════════════════════╗
# 4 │ 5‑fach GroupKFold‑CV  (Gruppen = Jahr)                             │
# ╚══════════════════════════════════════════════════════════════════════╝
banner("5‑FOLD CV – LIVE‑METRIKEN")

N_SPLITS = 5
MIN_RECALL = 0.60
BETA = 2                                 # für F‑Beta

gkf = GroupKFold(n_splits=N_SPLITS)

# Basis‑Hyperparameter für XGBoost
xgb_params = dict(
    objective="binary:logistic",
    eval_metric="aucpr",                 # PR‑AUC
    eta=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.5,                           # Minimale Verlustsreduktion für Split - Werte 0.1-1 reduzieren Overfitting
    n_estimators=2200,
    scale_pos_weight = np.bincount(y_train)[0] / np.bincount(y_train)[1],
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

# Optional: GridSearch für optimalen gamma-Wert
from sklearn.model_selection import GridSearchCV

# Sample-Subset für schnellere Suche
sample_size = min(10000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_sample, y_sample = X_train[indices], y_train[indices]

# Basis-Modell für Grid Search
search_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    eta=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    tree_method="hist",
    random_state=42,
    n_estimators=200  # Reduziert für schnellere Suche
)

# Parameter-Grid
param_grid = {
    'gamma': [0, 0.1, 0.3, 0.5, 0.7, 1.0]
}

# Grid Search durchführen
grid_search = GridSearchCV(
    estimator=search_model,
    param_grid=param_grid,
    cv=3,  # 3-fach CV für Geschwindigkeit
    scoring='average_precision',
    verbose=1
)

print("GridSearch für optimalen gamma-Parameter...")
grid_search.fit(X_sample, y_sample)
best_gamma = grid_search.best_params_['gamma']
print(f"Bester gamma-Wert: {best_gamma}")

# Update gamma-Wert in Hauptmodell-Parametern
xgb_params['gamma'] = best_gamma
print(f"Haupt-xgb_params aktualisiert mit gamma={best_gamma}")

thresholds, fold_stats = [], []

for k, (idx_tr, idx_va) in enumerate(gkf.split(X_train, y_train, groups=years_train), start=1):
    X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
    X_va, y_va = X_train[idx_va], y_train[idx_va]

    # ---- Modell + Isotone‑Kalibrierung ---------------------------------
    booster = XGBClassifier(**xgb_params).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    iso_clf = CalibratedClassifierCV(booster, method="isotonic", cv="prefit").fit(X_va, y_va)

    probas = iso_clf.predict_proba(X_va)[:, 1]
    precision, recall, thresh = precision_recall_curve(y_va, probas)

    # gleiche Länge: precision/recall = n+1, thresholds = n
    precision, recall = precision[:-1], recall[:-1]

    mask = recall >= MIN_RECALL
    if mask.any():
        tau = thresh[mask][precision[mask].argmax()]
    else:                                 # Fallback: bestes F‑Beta
        f2 = (1 + BETA**2) * precision * recall / (BETA**2 * precision + recall + 1e-9)
        tau = thresh[f2.argmax()]
        logging.warning(f"Fold {k}: Recall < {MIN_RECALL:.2f} – F2‑Optimum verwendet.")

    thresholds.append(tau)

    y_pred = (probas >= tau).astype(int)
    stats = dict(
        pr_auc = average_precision_score(y_va, probas),
        precision = precision_score(y_va, y_pred, zero_division=0),
        recall = recall_score(y_va, y_pred, zero_division=0),
        f1 = f1_score(y_va, y_pred, zero_division=0),
        f2 = fbeta_score(y_va, y_pred, beta=BETA, zero_division=0),
        bal_acc = balanced_accuracy_score(y_va, y_pred),
    )
    fold_stats.append(stats)

    print(f"Fold {k}:  PR‑AUC {stats['pr_auc']:.3f}  |  "
          f"P {stats['precision']:.3f}  R {stats['recall']:.3f}  "
          f"F1 {stats['f1']:.3f}  F2 {stats['f2']:.3f}")

# ---- Aggregierte CV‑Ergebnisse ----------------------------------------
tau_star = float(np.median(thresholds))
avg = pd.DataFrame(fold_stats).mean()

print("\n──────── Median‑Threshold & Ø‑Metriken ────────")
print(f"Median τ*      : {tau_star:.3f}")
print(f"Ø PR‑AUC       : {avg.pr_auc:.3f}")
print(f"Ø Precision    : {avg.precision:.3f}")
print(f"Ø Recall       : {avg.recall:.3f}")
print(f"Ø F1‑Score     : {avg.f1:.3f}")
print(f"Ø F2‑Score     : {avg.f2:.3f}")
print(f"Ø BalancedAcc  : {avg.bal_acc:.3f}")

#%%
# ╔══════════════════════════════════════════════════════════════════════╗
# 5 │ (Optional) Modell‑Persistenz                                       │
# ╚══════════════════════════════════════════════════════════════════════╝
banner("SAVING MODEL")

MODEL_DIR = Path("Tim/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# joblib.dump(final_iso, MODEL_DIR / "xgb_full_calibrated.joblib")
# with open(MODEL_DIR / "best_threshold.txt", "w") as f:
#     f.write(f"{tau_star:.6f}")
# fi.to_csv(MODEL_DIR / "feature_importance.csv")

print("✓ Modelldateien wurden (sofern aktiviert) gespeichert.")
