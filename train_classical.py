import json
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

from metrics import find_best_threshold, evaluate, log_result

DATA_DIR = "dataset_high"
RESULTS_PATH = "results.csv"
SEEDS = [0, 1, 2, 3, 4]
TUNE_SEEDS = [0, 1, 2]

PARAMS = {"lr": {"C": 10, "class_weight": None, "max_iter": 2000, "tol": 1e-8}, 
    "rf": {                          # cell I1/I2 -> tune_rf.csv, quy tac 1-SE
        "n_estimators": 300,
        "min_samples_leaf": 5,     # <- dan tu cell I2
        "max_features": 0.3,         # <- dan tu cell I2
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "xgb": {                         # dựa theo cell D/E/F -> tune_xgb_confirm.csv
        "max_depth": 13,
        "learning_rate": 0.0435,
        "reg_lambda": 0.02,
        "scale_pos_weight": 4.415,
        "colsample_bytree": 0.663,
        "subsample": 0.785,
        "n_estimators": 1000,
        "early_stopping_rounds": 20,
        "tree_method": "hist",
        "n_jobs": -1,
    },
    "lightGBM": {
    "num_leaves": 256,
    "learning_rate": 0.018,
    "n_estimators": 1000,
    "reg_lambda": 1.0,
    "reg_alpha": 1.8,
    "scale_pos_weight": 835.4312267657992,
    "max_bin": 63,
    "n_jobs": -1,
    "verbose": -1
},
    "mlp":{
    "hidden_layer_sizes": [32],
    "learning_rate_init": 0.003,
    "alpha": 0.0001,
    "batch_size": 2048,
    "max_iter": 40,
    "early_stopping": False,
    "solver": "adam",
    "activation": "relu"
    }
}  

def build_model(name, seed, pos_weight, cfg=None):
    p = cfg if cfg is not None else PARAMS[name]
    if name == "lr":
        return LogisticRegression(**p, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(**p, random_state=seed)
    if name == "xgb":
        if "scale_pos_weight" not in p:  
            p = {**p, "scale_pos_weight": pos_weight}
        return XGBClassifier(**p, random_state=seed, eval_metric="aucpr")

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
    if len(GRIDS[name]) == 1:        # xgb: params cứng từ analysis_1.ipynb, không tune
        return GRIDS[name][0]
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

def pick_best_model(path=RESULTS_PATH):
    d = pd.read_csv(path)
    d = d[(d.stage == "final") & (d.split == "test") & (d.model.isin(PARAMS))]
    d = d.drop_duplicates(subset=["model", "seed"], keep="last")
    means = d.groupby("model").f1_minority.mean()
    print("[final] f1_minority test trung binh:")
    for name, mu in means.sort_values(ascending=False).items():
        print(f"  {name}: {mu:.4f}")
    return str(means.idxmax()), float(means.max())

def load_split(split):
    df = pd.read_parquet(f"{DATA_DIR}/txn_matrix_{split}.parquet")
    y = df.pop("Is Laundering").to_numpy()
    return df.to_numpy(dtype="float32"), y, list(df.columns)


def main(stage):
    os.makedirs("scores", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    scaled = {"lr"}
    pos_weight = (y_train == 0).sum() / y_train.sum()

    if stage in ("tune", "all"):
        for name in tqdm(GRIDS, desc="tune models"):
            if len(GRIDS[name]) == 1:      # params cung, khong co gi de so sanh
                continue
            Xtr = X_train_s if name in scaled else X_train
            Xv = X_val_s if name in scaled else X_val
            tune(name, Xtr, y_train, Xv, y_val, pos_weight)
        if stage == "tune":                # dung lai de chay cell kiem tra bien
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

    best, best_f1 = pick_best_model()
    print(f"[best] {best} (f1_test={best_f1:.4f}) -> models/")

    Xbest = X_train_s if best in scaled else X_train
    for seed in tqdm(SEEDS, desc=f"save {best}"):
        model = build_model(best, seed, pos_weight)
        fit(model, best, Xbest, y_train)
        if best == "xgb":
            model.save_model(f"models/xgboost_seed{seed}.json")
        else:
            joblib.dump(model, f"models/{best}_seed{seed}.joblib")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert stage in ("tune", "final", "all"), \
        f"stage khong hop le: {stage!r} (chon: tune | final | all)"
    main(stage)