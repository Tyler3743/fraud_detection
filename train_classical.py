"""Nhom 1 — classical baselines per-transaction (plan.md buoc 5).

LR / DT / RF / XGBoost / MLP tren ma tran assemble_txn.
Chay 5 seed/model, threshold tune tren VAL, log vao results.csv.
Xuat: scores/{model}_seed{s}_{split}.npy, models/xgboost_seed{s}.json (V0).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from metrics import find_best_threshold, evaluate, log_result

DATA_DIR = "dataset_high"
SEEDS = [0, 1, 2, 3, 4]

# Hyperparameter co dinh — ghi JSON vao results.csv lam bang phu luc.
PARAMS = {
    "lr":  {"max_iter": 1000, "class_weight": "balanced"},
    "dt":  {"max_depth": 10, "class_weight": "balanced"},
    "rf":  {"n_estimators": 100, "max_depth": 12,
            "class_weight": "balanced", "n_jobs": -1},
    "xgb": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1,
            "tree_method": "hist", "n_jobs": -1},  # scale_pos_weight them luc build
    "mlp": {"hidden_layer_sizes": (64, 32), "max_iter": 30,
            "early_stopping": True},
}


def build_model(name, seed, pos_weight):
    p = PARAMS[name]
    if name == "lr":
        return LogisticRegression(**p, random_state=seed)
    if name == "dt":
        return DecisionTreeClassifier(**p, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(**p, random_state=seed)
    if name == "xgb":
        return XGBClassifier(**p, scale_pos_weight=pos_weight,
                             random_state=seed, eval_metric="aucpr")
    if name == "mlp":
        return MLPClassifier(**p, random_state=seed)


def load_split(split):
    df = pd.read_parquet(f"{DATA_DIR}/txn_matrix_{split}.parquet")
    y = df.pop("Is Laundering").to_numpy()
    return df.to_numpy(dtype="float32"), y, list(df.columns)


def main():
    os.makedirs("scores", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train, y_train, feat_names = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    # Scaler fit TRAIN-ONLY (plan.md muc 6 — chong leakage).
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pos_weight = (y_train == 0).sum() / y_train.sum()

    for name in PARAMS:
        for seed in SEEDS:
            model = build_model(name, seed, pos_weight)
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time_s = round(time.time() - t0, 1)

            s_val = model.predict_proba(X_val)[:, 1]
            s_test = model.predict_proba(X_test)[:, 1]
            np.save(f"scores/{name}_seed{seed}_val.npy", s_val)
            np.save(f"scores/{name}_seed{seed}_test.npy", s_test)

            thr = find_best_threshold(y_val, s_val)  # tune tren VAL
            for split, y, s in [("val", y_val, s_val), ("test", y_test, s_test)]:
                log_result(name, split, evaluate(y, s, thr),
                           seed=seed, train_time_s=train_time_s,
                           params=json.dumps(PARAMS[name]))

            if name == "xgb":  # V0 — tai dung o nhom 3 + SHAP buoc 8
                model.save_model(f"models/xgboost_seed{seed}.json")
                imp = pd.Series(model.feature_importances_, index=feat_names)
                imp.sort_values(ascending=False).to_csv(
                    f"models/xgboost_seed{seed}_importance.csv")


if __name__ == "__main__":
    main()