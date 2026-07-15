"""Do do chung cap giao dich cho ca 3 nhom model (plan.md muc 6).

Metric chinh: minority-F1 (so voi moc 63.23 / 68.16).
Phu: PR-AUC, recall@FPR, precision@k. Threshold tune tren VAL, ap sang TEST.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve, roc_curve,
)

RESULTS_PATH = "results.csv"


def find_best_threshold(y_true, y_score):
    """Threshold cho minority-F1 lon nhat (tune tren val)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best = np.argmax(f1[:-1])  # phan tu cuoi khong co threshold tuong ung
    return float(thr[best])


def recall_at_fpr(y_true, y_score, max_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(max_fpr, fpr, tpr))


def precision_at_k(y_true, y_score, k=1000):
    order = np.argsort(y_score)[::-1][:k]
    return float(np.asarray(y_true)[order].mean())


def evaluate(y_true, y_score, threshold=0.5):
    """Tra ve dict metric cap giao dich. y_score = xac suat/diem lop duong."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "f1_minority": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "recall@fpr1%": recall_at_fpr(y_true, y_score, 0.01),
        "precision@1000": precision_at_k(y_true, y_score, 1000),
        "threshold": threshold,
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
    }


def evaluate_val_test(y_val, s_val, y_test, s_test):
    """Quy trinh chuan: tune threshold tren val -> danh gia ca val va test."""
    thr = find_best_threshold(y_val, s_val)
    return {
        "val": evaluate(y_val, s_val, thr),
        "test": evaluate(y_test, s_test, thr),
    }


def log_result(model, split, metrics, path=RESULTS_PATH, **extra):
    """Append 1 dong vao results.csv. Ghi ro split (plan.md muc 6)."""
    row = {"model": model, "split": split,
           "time": pd.Timestamp.now().isoformat(timespec="seconds"),
           **metrics, **extra}
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))
    print(f"[{model} | {split}] f1_minority={metrics['f1_minority']:.4f} "
          f"pr_auc={metrics['pr_auc']:.4f} thr={metrics['threshold']:.4f}")