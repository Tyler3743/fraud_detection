import json
import os
import time
from tqdm import tqdm
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

PARAMS = {
    "lr":  {"max_iter": 1000, "class_weight": "balanced"},
    "dt":  {"max_depth": 10, "class_weight": "balanced"},
    "rf":  {"n_estimators": 100, "max_depth": 12,
            "class_weight": "balanced", "n_jobs": -1},
    "xgb": {"n_estimators": 1000, "max_depth": 8, "learning_rate": 0.1,
            "tree_method": "hist", "n_jobs": -1, "early_stopping_rounds": 20,"subsample": 0.8, "colsample_bytree": 0.8},
    "mlp": {"hidden_layer_sizes": (64, 32), "max_iter": 100,
            "early_stopping": True},
}

GRIDS = {
    "xgb": [{"max_depth": d, "learning_rate": lr, "scale_pos_weight": spw,
             "n_estimators": 1000, "early_stopping_rounds": 20,"subsample": 0.8, "colsample_bytree": 0.8,
             "tree_method": "hist", "n_jobs": -1}
            for d in [6, 8, 10] for lr in [0.05, 0.1] for spw in [1, 10, 100]],
    "rf":  [{"n_estimators": n, "max_depth": d, "class_weight": cw, "n_jobs": -1}
            for n in [100, 300] for d in [16, None]
            for cw in ["balanced", "balanced_subsample"]],
    "dt":  [{"max_depth": d, "class_weight": cw}
            for d in [8, 12, 16] for cw in ["balanced", None]],
}


def build_model(name, seed, pos_weight, cfg=None):
    p = cfg if cfg is not None else PARAMS[name]
    if name == "lr":
        return LogisticRegression(**p, random_state=seed)
    if name == "dt":
        return DecisionTreeClassifier(**p, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(**p, random_state=seed)
    if name == "xgb":
        if "scale_pos_weight" not in p:  # config goc: dung full ratio
            p = {**p, "scale_pos_weight": pos_weight}
        return XGBClassifier(**p, random_state=seed, eval_metric="aucpr")
    if name == "mlp":
        return MLPClassifier(**p, random_state=seed)

ES_FRAC = 0.1  # đuôi train (theo thời gian) dành riêng cho xgb early-stopping, KHÔNG dùng val


def fit(model, name, Xtr, y_train):
    if name == "xgb":   
        cut = int(len(y_train) * (1 - ES_FRAC))
        Xfit, yfit = Xtr[:cut], y_train[:cut]
        Xes, yes = Xtr[cut:], y_train[cut:]
        model.fit(Xfit, yfit, eval_set=[(Xes, yes)])
    else:
        model.fit(Xtr, y_train)


def tune(name, Xtr, y_train, Xv, y_val, pos_weight):
    best, best_f1 = None, -1
    for cfg in tqdm(GRIDS[name], desc=f"tune {name}"):
        model = build_model(name, 0, pos_weight, cfg)  # seed 0
        fit(model, name, Xtr, y_train)  # <- bo Xv, y_val: xgb early-stopping dung duoi train, khong dung val
        s_val = model.predict_proba(Xv)[:, 1]
        m = evaluate(y_val, s_val, find_best_threshold(y_val, s_val))
        log_result(name, "val", m, seed=0, stage="tune",
                   params=json.dumps(cfg))
        if m["f1_minority"] > best_f1:
            best, best_f1 = cfg, m["f1_minority"]
    return best



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

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    scaled = {"lr", "mlp"}
    pos_weight = (y_train == 0).sum() / y_train.sum()

    for name in tqdm(GRIDS, desc="tune models"):  # buoc tune: ghi de PARAMS bang best config
        Xtr = X_train_s if name in scaled else X_train
        Xv = X_val_s if name in scaled else X_val
        PARAMS[name] = tune(name, Xtr, y_train, Xv, y_val, pos_weight)
        print(f"[tune] best {name}: {PARAMS[name]}")

    for name in tqdm(PARAMS, desc="train models"):  # buoc final: 5 seed
        Xtr = X_train_s if name in scaled else X_train
        Xv = X_val_s if name in scaled else X_val
        Xt = X_test_s if name in scaled else X_test
        for seed in tqdm(SEEDS, desc=f"{name} seeds", leave=False):
            model = build_model(name, seed, pos_weight)
            t0 = time.time()
            fit(model, name, Xtr, y_train)
            train_time_s = round(time.time() - t0, 1)

            s_val = model.predict_proba(Xv)[:, 1]
            s_test = model.predict_proba(Xt)[:, 1]
            np.save(f"scores/{name}_seed{seed}_val.npy", s_val)
            np.save(f"scores/{name}_seed{seed}_test.npy", s_test)

            thr = find_best_threshold(y_val, s_val)
            for split, y, s in [("val", y_val, s_val), ("test", y_test, s_test)]:
                log_result(name, split, evaluate(y, s, thr),
                           seed=seed, train_time_s=train_time_s,
                           stage="final", params=json.dumps(PARAMS[name]))

            if name == "xgb":  # V0 — tai dung o nhom 3 + SHAP buoc 8
                model.save_model(f"models/xgboost_seed{seed}.json")
                imp = pd.Series(model.feature_importances_, index=feat_names)
                imp.sort_values(ascending=False).to_csv(
                    f"models/xgboost_seed{seed}_importance.csv")


if __name__ == "__main__":
    main()