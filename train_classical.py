import json
import os
import sys
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
RESULTS_PATH = "results.csv"
SEEDS = [0, 1, 2, 3, 4]
TUNE_SEEDS = [0, 1, 2]

PARAMS = {"lr": {}, "dt": {}, "rf": {}, "xgb": {}, "mlp": {}}  
GRIDS = {
    "lr":  [{"C": c, "class_weight": None, "max_iter": 1000}
            for c in [0.01, 0.1, 1.0]],

    "dt":  [{"max_depth": d, "class_weight": "balanced"}
            for d in [12, 16, 20]],

    "rf":  [{"n_estimators": n, "max_depth": None, "min_samples_leaf": l,
             "class_weight": "balanced", "n_jobs": -1}
            for n in [100, 300] for l in [20, 5, 1]],

    "xgb": [{"max_depth": 8, "learning_rate": 0.05,
             "scale_pos_weight": spw, "reg_lambda": lam,
             "n_estimators": 1000, "early_stopping_rounds": 20,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "tree_method": "hist", "n_jobs": -1}
            for lam in [10.0, 1.0,0.1] for spw in [1,5,10]],

    "mlp": [{"hidden_layer_sizes": h, "alpha": 1e-4, "batch_size": 512,
             "max_iter": 30, "early_stopping": False}
            for h in [(64, 32), (128, 64)]],
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
    for cfg in tqdm(GRIDS[name], desc=f"tune {name}"):
        for seed in TUNE_SEEDS:
            model = build_model(name, seed, pos_weight, cfg)
            t0 = time.time()
            fit(model, name, Xtr, y_train)
            train_time_s = round(time.time() - t0, 1)
            s_val = model.predict_proba(Xv)[:, 1]
            m = evaluate(y_val, s_val, find_best_threshold(y_val, s_val))
            log_result(name, "val", m, seed=seed, stage="tune",
                       train_time_s=train_time_s, params=json.dumps(cfg))


def pick_best(name, path=RESULTS_PATH):
    d = pd.read_csv(path)
    d = d[(d.model == name) & (d.stage == "tune") & (d.split == "val")]
    d = d.drop_duplicates(subset=["params", "seed"], keep="last")
    g = d.groupby("params").f1_minority

    means, ses = [], []
    for cfg in GRIDS[name]:
        f1s = g.get_group(json.dumps(cfg))
        means.append(float(f1s.mean()))
        ses.append(float(f1s.std(ddof=1) / np.sqrt(len(f1s))))

    b = int(np.argmax(means))
    cutoff = means[b] - ses[b]
    return GRIDS[name][next(i for i, mu in enumerate(means) if mu >= cutoff)]


def load_split(split):
    df = pd.read_parquet(f"{DATA_DIR}/txn_matrix_{split}.parquet")
    y = df.pop("Is Laundering").to_numpy()
    return df.to_numpy(dtype="float32"), y, list(df.columns)


def main(stage):
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

    if stage == "tune":  # chi ghi log -> dung lai de chay cell kiem tra bien
        for name in tqdm(GRIDS, desc="tune models"):
            Xtr = X_train_s if name in scaled else X_train
            Xv = X_val_s if name in scaled else X_val
            tune(name, Xtr, y_train, Xv, y_val, pos_weight)
        return

    for name in tqdm(PARAMS, desc="train models"):  
        PARAMS[name] = pick_best(name)
        print(f"[final] best {name}: {PARAMS[name]}")
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

            if name == "xgb":  # V0 — tái dùng ở bước, SHAP bước 8
                model.save_model(f"models/xgboost_seed{seed}.json")
                imp = pd.Series(model.feature_importances_, index=feat_names)
                imp.sort_values(ascending=False).to_csv(
                    f"models/xgboost_seed{seed}_importance.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tune")