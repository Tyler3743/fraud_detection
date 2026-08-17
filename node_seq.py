import gc
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_node import load_data
from feature_edge import add_subsplit, WINDOWS
from paths import OUT_DIR, TXN_NODES

BUCKET_H = 6            # độ mịn bucket (giờ)
K = 4                   # giữ K bucket CUỐI của cửa sổ nguồn = 1 ngày, căn phải
NS = 1_000_000_000
COLS = ["out_cnt", "out_sum", "out_nuniq", "in_cnt", "in_sum", "in_nuniq",
        "ratio_cross_bank", "ratio_cross_ccy"]
LOG_IDX = [0, 1, 2, 3, 4, 5]                 
NODE_SEQ = os.path.join(OUT_DIR, "node_seq_{}.npy")


def build_seq(df, src_splits, txn, n_node):
    """Chuỗi hoạt động của từng node trên K bucket cuối của cửa sổ NGUỒN. Không đụng nhãn."""
    w = df[df["win"].isin(src_splits)]
    b = w["time"].to_numpy() // (BUCKET_H * 3600 * NS)
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
    txn = np.load(TXN_NODES)                             # (n_tx, 2) theo THỨ TỰ FILE GỐC
    n_node = int(txn.max()) + 1
    print(f"{n_node:,} node | bucket {BUCKET_H}h | K={K} ({K*BUCKET_H/24:.1f} ngày cuối)")

    seqs = {n: build_seq(df, w[0], txn, n_node) for n, w in WINDOWS.items()}
    verify(df, seqs, txn, n_node)
    seqs = scale(seqs)
    for name, Z in seqs.items():
        p = NODE_SEQ.format(name)
        np.save(p, Z)
        print(f"  đã lưu {p} shape={Z.shape} ({Z.nbytes/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
