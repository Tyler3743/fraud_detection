import json
import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from metrics import find_best_threshold, evaluate, log_result
import matplotlib
matplotlib.use("Agg")          # không cần cửa sổ đồ họa, chỉ ghi file
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DATA_DIR = "dataset_high"
RESULTS_PATH = "results.csv"
SEEDS = [0, 1, 2, 3]
FIG_DIR = "figures"

PARAMS = {
    "rf": {                         
        "n_estimators": 300,
        "min_samples_leaf": 5,    
        "max_features": 0.3,         
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "xgb": {                        
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
    "lgb": {
    "num_leaves": 128,
    "learning_rate": 0.03443,
    "reg_lambda": 44.8638,
    "reg_alpha": 2.5893,
    "scale_pos_weight": 2.612,
    "n_estimators": 1000,
    "max_bin": 63,
    "n_jobs": -1,
    "verbose": -1
},
    "lr":{
    "class_weight": None, "C":10,  "max_iter":10000, "tol":1e-8 
    },
    "dt":{
    "min_samples_leaf": 25,
    "max_depth": 24,
    "class_weight": "balanced"
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
    if name == "lgb":
        if "scale_pos_weight" not in p:
            p = {**p, "scale_pos_weight": pos_weight}
        return lgb.LGBMClassifier(**p, random_state=seed)
    if name == "dt":
        return DecisionTreeClassifier(**p, random_state=seed)
    raise ValueError(f"model không tồn tại {name!r}")

ES_FRAC = 0.1  # đuôi train (theo thời gian) dành riêng cho xgb early-stopping, KHÔNG dùng val


def fit(model, name, Xtr, y_train):
    if name in ("xgb", "lgb"):                       
        cut = int(len(y_train) * (1 - ES_FRAC))
        Xfit, yfit = Xtr[:cut], y_train[:cut]
        Xes, yes = Xtr[cut:], y_train[cut:]
        if name == "xgb":
            model.fit(Xfit, yfit, eval_set=[(Xes, yes)])
        else:
            model.fit(Xfit, yfit, eval_set=[(Xes, yes)],
                      eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(20, first_metric_only=True, verbose=False),
                                 lgb.log_evaluation(0)])
    else:
        model.fit(Xtr, y_train)

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

def mean_score(name, split):
    return np.mean([np.load(f"scores/{name}_seed{s}_{split}.npy") for s in SEEDS], axis=0)

CM_TAGS = [["TN", "FP"], ["FN", "TP"]]


def draw_cm(ax, cm, title, fontsize=13):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
    for a in range(2):
        for b in range(2):
            ax.text(b, a,
                    f"{CM_TAGS[a][b]}\n{cm[a, b]:,.0f}\n({cmn[a, b]:.4f})",
                    ha="center", va="center", fontsize=fontsize, linespacing=1.7,
                    color="white" if cmn[a, b] > 0.5 else "black")
    ax.set_xticks([0, 1], ["dự đoán\nbình tường", "dự đoán\nrửa tiền"], fontsize=9)
    ax.set_yticks([0, 1], ["nhãn\nbình thường", "nhãn\nrửa tiền"], fontsize=9)
    ax.set_title(title, fontsize=11) 

def plot_confusion_matrices(y_val, y_test, names, path=None):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = path or f"{FIG_DIR}/confusion_matrix_test.png"
    fig, axes = plt.subplots(2, len(names), figsize=(4.0 * len(names), 8.0))
    axes = np.atleast_2d(axes).reshape(2, len(names))

    for j, name in enumerate(names):
        thr = find_best_threshold(y_val, mean_score(name, "val"))#dò ngưỡng
        y_pred = (mean_score(name, "test") >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cmn = cm / cm.sum(axis=1, keepdims=True)

        for r, (mat, fmt, tag) in enumerate([(cm, "{:,.0f}", "số đếm"),
                                             (cmn, "{:.4f}", "chuẩn hóa theo hàng")]):
            ax = axes[r, j]
            ax.imshow(cmn, cmap="Blues", vmin=0.0, vmax=1.0)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, fmt.format(mat[a, b]), ha="center", va="center",
                            fontsize=10,
                            color="white" if cmn[a, b] > 0.5 else "black")
            ax.set_xticks([0, 1], ["dự đoán\nhợp lệ", "dự đoán\nrửa tiền"], fontsize=8)
            ax.set_yticks([0, 1], ["thật\nhợp lệ", "thật\nrửa tiền"], fontsize=8)
            ax.set_title(f"{name} — {tag}" + (f"\nthr={thr:.4f}" if r == 0 else ""),
                         fontsize=10)
        fig1, ax1 = plt.subplots(figsize=(4.6, 4.6))
        draw_cm(ax1, cm, f"{name} — test (thr={thr:.4f})")
        p1 = f"{FIG_DIR}/confusion_matrix_test_{name}.png"
        fig1.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"[figure] đã lưu {p1}")

    fig.suptitle("Ma trận nhầm lẫn trên tập test (điểm trung bình 5 seed, "
                 "ngưỡng chọn trên val)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] đã lưu {path}")


def print_summary(rows):
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="model", columns="split", values="f1",
                        aggfunc=["mean", "std"])
    order = piv[("mean", "test")].sort_values(ascending=False).index

    print("\n" + "=" * 58)
    print(f"{'model':<8}{'f1 val (mean±std)':>25}{'f1 test (mean±std)':>25}")
    print("-" * 58)
    for m in order:
        print(f"{m:<8}"
              f"{piv.loc[m, ('mean','val')]:>16.4f} ± {piv.loc[m, ('std','val')]:.4f}"
              f"{piv.loc[m, ('mean','test')]:>16.4f} ± {piv.loc[m, ('std','test')]:.4f}")
    print("=" * 58)

def main():
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

    rows = []
    bar = tqdm(PARAMS, desc="models", unit="model")
    for name in bar:
        bar.set_postfix_str(name)
        tqdm.write(f"\n[{name}] {PARAMS[name]}")
        Xtr = X_train_s if name in scaled else X_train
        Xv = X_val_s if name in scaled else X_val
        Xt = X_test_s if name in scaled else X_test

        sbar = tqdm(SEEDS, desc=f"  {name} seeds", unit="seed", leave=False)
        for seed in sbar:
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
                m = evaluate(y, s, thr)#độ đo
                log_result(name, split, m, seed=seed, train_time_s=train_time_s,
                           stage="final", params=json.dumps(PARAMS[name]))
                rows.append({"model": name, "split": split, "seed": seed,
                             "f1": m["f1_minority"]})

            f1v = rows[-2]["f1"]
            f1t = rows[-1]["f1"]
            sbar.set_postfix_str(f"f1_val={f1v:.4f} f1_test={f1t:.4f} {train_time_s}s")
            tqdm.write(f"  seed {seed}: {train_time_s:>7.1f}s | "
                       f"f1_val={f1v:.4f} | f1_test={f1t:.4f} | thr={thr:.4f}")
        sbar.close()
    bar.close()

    print_summary(rows)
    plot_confusion_matrices(y_val, y_test, list(PARAMS))

    best, best_f1 = pick_best_model()
    print(f"\n[best] {best} (f1_test={best_f1:.4f}) -> models/")

    Xbest = X_train_s if best in scaled else X_train
    bbar = tqdm(SEEDS, desc=f"lưu {best}", unit="model")
    for seed in bbar:
        model = build_model(best, seed, pos_weight)
        t0 = time.time()
        fit(model, best, Xbest, y_train)
        if best == "xgb":
            out = f"models/xgboost_seed{seed}.json"
            model.save_model(out)
        else:
            out = f"models/{best}_seed{seed}.joblib"
            joblib.dump(model, out)
        dt = round(time.time() - t0, 1)
        bbar.set_postfix_str(f"seed{seed} {dt}s")
        tqdm.write(f"  [{seed + 1}/{len(SEEDS)}] {out}  ({dt}s, "
                   f"{os.path.getsize(out) / 1e6:.1f} MB)")
    bbar.close()

if __name__ == "__main__": main()