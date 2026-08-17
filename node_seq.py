import gc
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_node import load_data
from feature_edge import DELTA_H
from paths import OUT_DIR, TXN_NODES

BUCKET_MIN = 90                   
K = 4
NS = 1_000_000_000
BUCKET_NS = BUCKET_MIN * 60 * NS
COLS = ["out_cnt", "out_sum", "out_nuniq", "in_cnt", "in_sum", "in_nuniq",
        "ratio_cross_bank", "ratio_cross_ccy"]
LOG_IDX = [0, 1, 2, 3, 4, 5]
NODE_SEQ = os.path.join(OUT_DIR, "node_seq_{}.npy")
TRAIN_B_MASK = os.path.join(OUT_DIR, "train_b_mask.npy")

WINDOWS = {
    "train": (["train_a"],                    ["train_a", "train_b"]),
    "val":   (["train_a", "train_b"],         ["train_a", "train_b", "val"]),
    "test":  (["train_a", "train_b", "val"],  ["train_a", "train_b", "val", "test"]),
}


def add_subsplit(df):
    df["win"] = df["split"]
    tr = (df["split"] == "train").to_numpy()
    ts_cut = df.loc[tr, "Timestamp"].min() + pd.Timedelta(hours=DELTA_H)

    is_b = tr & (df["Timestamp"] >= ts_cut).to_numpy()
    df.loc[is_b, "win"] = "train_b"
    df.loc[tr & ~is_b, "win"] = "train_a"

    a, b = df[df.win == "train_a"], df[df.win == "train_b"]
    assert len(a) and len(b), "train_a hoặc train_b rỗng"
    assert a["Timestamp"].max() < b["Timestamp"].min(), "train_a/train_b chồng lấn thời gian"
    print(f"  cắt train tại {ts_cut} ({DELTA_H}h đầu) | train_a {len(a):,} dòng "
          f"(pos {int(a['Is Laundering'].sum()):,}) | train_b {len(b):,} dòng "
          f"(pos {int(b['Is Laundering'].sum()):,})")

    mask = np.zeros(len(df), dtype=bool)
    mask[df["ori_idx"].to_numpy()] = is_b
    np.save(TRAIN_B_MASK, mask)
    print(f"  đã lưu {TRAIN_B_MASK}")
    return df


def build_seq(df, src_splits, txn, n_node):
    w = df[df["win"].isin(src_splits)]
    b = w["time"].to_numpy("int64") // BUCKET_NS
    pos = b - (b.max() - K + 1)
    keep = pos >= 0
    w, pos = w[keep], pos[keep]

    o = w["ori_idx"].to_numpy()
    sid, did = txn[o, 0], txn[o, 1]
    Z = np.zeros((n_node, K, len(COLS)), dtype="float32")

    # vai GỬI
    t = pd.DataFrame({"n": sid, "p": pos, "amt": w["Amount Paid"].to_numpy("float64"),
                      "partner": did,
                      "cb": w["is_cross_bank"].to_numpy("float32"),
                      "cc": w["is_cross_curcy"].to_numpy("float32")})
    a = t.groupby(["n", "p"], sort=False).agg(
        cnt=("amt", "size"), s=("amt", "sum"), nuniq=("partner", "nunique"),
        cb=("cb", "mean"), cc=("cc", "mean"))
    i, j = a.index.get_level_values(0).to_numpy(), a.index.get_level_values(1).to_numpy()
    Z[i, j, 0] = a["cnt"]; Z[i, j, 1] = a["s"]; Z[i, j, 2] = a["nuniq"]
    Z[i, j, 6] = a["cb"];  Z[i, j, 7] = a["cc"]
    del t, a; gc.collect()

    # vai NHẬN
    t = pd.DataFrame({"n": did, "p": pos, "amt": w["Amount Received"].to_numpy("float64"),
                      "partner": sid})
    a = t.groupby(["n", "p"], sort=False).agg(
        cnt=("amt", "size"), s=("amt", "sum"), nuniq=("partner", "nunique"))
    i, j = a.index.get_level_values(0).to_numpy(), a.index.get_level_values(1).to_numpy()
    Z[i, j, 3] = a["cnt"]; Z[i, j, 4] = a["s"]; Z[i, j, 5] = a["nuniq"]
    del t, a; gc.collect()
    return Z


def verify(df, seqs, txn, n_node):
    for name, target in [("train", "train_b"), ("val", "val")]:
        s2 = df.copy()
        m = (s2["win"] == target).to_numpy()
        rng = np.random.default_rng(0)
        for col in ["Amount Paid", "Amount Received"]:
            s2.loc[m, col] = rng.permutation(s2.loc[m, col].to_numpy())
        ref = build_seq(s2, WINDOWS[name][0], txn, n_node)
        assert np.array_equal(ref, seqs[name]), f"node_seq {name} ĐANG phụ thuộc {target} -> leak"
        print(f"  PASS: node_seq {name} bất biến khi xáo dữ liệu {target}")
        del s2, ref; gc.collect()

    for name, Z in seqs.items():
        act = (Z.reshape(n_node, -1) != 0).any(1).mean()
        print(f"  {name:5s}: {Z.shape} | node hoạt động {act*100:5.2f}%")


def scale(seqs):
    for Z in seqs.values():
        Z[:, :, LOG_IDX] = np.log1p(np.clip(Z[:, :, LOG_IDX], 0, None))
    sc = StandardScaler().fit(seqs["train"].reshape(-1, len(COLS))[:, LOG_IDX])
    for Z in seqs.values():
        f = Z.reshape(-1, len(COLS))                     # view, ghi ngược lại Z
        f[:, LOG_IDX] = sc.transform(f[:, LOG_IDX]).astype("float32")
    return seqs


def main():
    df = add_subsplit(load_data())
    txn = np.load(TXN_NODES)                             
    n_node = int(txn.max()) + 1
    print(f"{n_node:,} node | bucket {BUCKET_MIN} phút | K={K} "
          f"({K*BUCKET_MIN/60:.1f}h cuối, khớp DELTA_H={DELTA_H}h)")

    seqs = {n: build_seq(df, w[0], txn, n_node) for n, w in WINDOWS.items()}
    verify(df, seqs, txn, n_node)
    seqs = scale(seqs)
    for name, Z in seqs.items():
        p = NODE_SEQ.format(name)
        np.save(p, Z)
        print(f"  đã lưu {p} shape={Z.shape} ({Z.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
