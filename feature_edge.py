import os
import gc
import numpy as np
import pandas as pd

from feature_node import load_data      # đã sort Timestamp + tạo src/dest + cờ
from paths import EDGE_ATTR, OUT_DIR

EDGE_RANK_COLS = [
    "num_tx", "total_paid", "mean_paid", "std_paid",
    "min_paid", "max_paid", "active_hour", "tx_per_hour",
    "port_out", "port_in",
]

TRAIN_A_FRAC = 0.5                                      
TRAIN_B_MASK = os.path.join(OUT_DIR, "train_b_mask.npy")

WINDOWS = {
    "train": (["train_a"],                    ["train_a", "train_b"]),
    "val":   (["train_a", "train_b"],         ["train_a", "train_b", "val"]),
    "test":  (["train_a", "train_b", "val"],  ["train_a", "train_b", "val", "test"]),
}


def add_subsplit(df):
    """Cắt cửa sổ train theo THỜI GIAN: train_a (tính feature) / train_b (nhãn để train)."""
    df["win"] = df["split"]
    tr = np.flatnonzero((df["split"] == "train").to_numpy())
    ts_cut = df["Timestamp"].iloc[tr[int(len(tr) * TRAIN_A_FRAC)]]

    is_b = (df["split"] == "train") & (df["Timestamp"] >= ts_cut)
    df.loc[is_b, "win"] = "train_b"
    df.loc[(df["split"] == "train") & ~is_b, "win"] = "train_a"

    a, b = df[df.win == "train_a"], df[df.win == "train_b"]
    assert len(a) and len(b), "train_a hoặc train_b rỗng"
    assert a["Timestamp"].max() < b["Timestamp"].min(), "train_a/train_b chồng lấn thời gian"
    print(f"  cắt train tại {ts_cut} | train_a {len(a):,} dòng "
          f"(pos {int(a['Is Laundering'].sum()):,}) | train_b {len(b):,} dòng "
          f"(pos {int(b['Is Laundering'].sum()):,})")

    # mask theo THỨ TỰ FILE GỐC để train_gnn.py ghép đúng dòng
    mask = np.zeros(len(df), dtype=bool)
    mask[df["ori_idx"].to_numpy()] = is_b.to_numpy()
    np.save(TRAIN_B_MASK, mask)
    print(f"  đã lưu {TRAIN_B_MASK}")
    return df

def aggregate_edges(win: pd.DataFrame, formats, currencies):
    """Thống kê cạnh gộp trên cửa sổ NGUỒN. Không đụng tới 'Is Laundering'."""
    g = win.groupby(["src", "dest"], sort=False)
    agg = g.agg(
        num_tx          = ("Amount Paid", "size"),
        total_paid      = ("Amount Paid", "sum"),
        mean_paid       = ("Amount Paid", "mean"),
        std_paid        = ("Amount Paid", "std"),
        min_paid        = ("Amount Paid", "min"),
        max_paid        = ("Amount Paid", "max"),
        cross_ccy_ratio = ("is_cross_curcy", "mean"),
        time_min        = ("Timestamp", "min"),
        time_max        = ("Timestamp", "max"),
    )
    agg["std_paid"] = agg["std_paid"].fillna(0.0)         
    span_h = (agg["time_max"] - agg["time_min"]).dt.total_seconds() / 3600.0
    agg["active_hour"] = span_h.clip(lower=1.0)
    agg["tx_per_hour"] = agg["num_tx"] / agg["active_hour"]

    # Đánh số cổng theo mốc thời gian (Multi-GNN mục 4.2): mọi cạnh tới cùng một láng
    # giềng nhận cùng số cổng; thứ tự do timestamp SỚM NHẤT của cặp quyết định. Tính
    # trên cửa sổ NGUỒN nên vẫn trễ pha, không đụng dữ liệu của kỳ được gán nhãn.
    tmin = agg["time_min"].astype("int64")
    agg["port_out"] = tmin.groupby(level="src").rank(method="first")
    agg["port_in"] = tmin.groupby(level="dest").rank(method="first")

    agg = agg.drop(columns=["time_min", "time_max"])

    keys = [win["src"], win["dest"]]
    for col, names, prefix in [("Payment Format", formats, "pf"),
                               ("Payment Currency", currencies, "ccy")]:
        d = pd.get_dummies(win[col]).astype("float32").reindex(columns=names, fill_value=0.0)
        prop = d.groupby(keys, sort=False).mean()
        prop.columns = [f"{prefix}_{c.replace(' ', '_')}" for c in names]
        agg = agg.join(prop)
    return agg


def build_window(df, src_splits, graph_splits, formats, currencies):
    s_window = df[df["win"].isin(src_splits)]      
    g_window = df[df["win"].isin(graph_splits)]    
    edges = g_window.groupby(["src", "dest"], sort=False).agg(
        is_cross_bank = ("is_cross_bank", "max"),
        is_self_loop  = ("is_self_loop",  "max"),
    ).astype("int8")

    agg = aggregate_edges(s_window, formats, currencies)

    feat = edges.join(agg, how="left")
    feat["seen_before"] = feat["num_tx"].notna().astype("int8")
    return feat.fillna(0.0)


def rank_edges(feats, cols):
    for name, f in feats.items():
        r = f[cols].astype("float64")
        r[f["seen_before"] == 0] = np.nan       
        f[cols] = r.rank(pct=True).fillna(0.0)  
        feats[name] = f
    return feats

def verify(df, feats, formats, currencies):
    # 1. không có cột nào suy từ nhãn
    banned = {"is_edge_mule", "is_mule", "Is Laundering"}
    for name, f in feats.items():
        assert not (banned & set(f.columns)), f"{name} còn cột suy từ nhãn"

    for name, target in [("train", "train_b"), ("val", "val")]:
        shuffled = df.copy()
        m = shuffled["win"] == target
        assert m.sum() > 0, f"không có dòng nào thuộc {target} -> add_subsplit chưa chạy"
        rng = np.random.default_rng(0)
        shuffled.loc[m, "Amount Paid"] = rng.permutation(shuffled.loc[m, "Amount Paid"].values)
        ref = build_window(shuffled, *WINDOWS[name], formats, currencies)
        assert ref[list(feats[name].columns)].equals(feats[name]), \
            f"edge_attr {name} ĐANG phụ thuộc {target} -> leak"
        print(f"  PASS: edge_attr {name} bất biến khi xáo dữ liệu {target}")
        del shuffled, ref
        gc.collect()

    for name, f in feats.items():
        r = 1 - f["seen_before"].mean()
        print(f"  {name:5s}: {len(f):>9,} cạnh | cold-start {r*100:5.2f}%")
    cc = {n: feats[n].loc[feats[n]["seen_before"] == 1, EDGE_RANK_COLS]
                     .corr().abs().fillna(0.0) for n in ("train", "test")}
    gap = (cc["train"] - cc["test"]).abs().to_numpy(copy=True)
    np.fill_diagonal(gap, 0.0)
    bad = (cc["train"].to_numpy() > 0.99) & (gap > 0.05)
    np.fill_diagonal(bad, False)
    assert not bad.any(), "cột trùng ở train nhưng tách ở test: " + ", ".join(
        f"{EDGE_RANK_COLS[a]}~{EDGE_RANK_COLS[b]} "
        f"(train {cc['train'].iloc[a, b]:.4f} -> test {cc['test'].iloc[a, b]:.4f})"
        for a, b in zip(*np.where(np.triu(bad, 1))))
    i, j = np.unravel_index(gap.argmax(), gap.shape)
    print(f"  PASS: chênh |corr| train-test lớn nhất = {gap[i, j]:.4f} "
          f"({EDGE_RANK_COLS[i]} ~ {EDGE_RANK_COLS[j]})")


def main():
    df = load_data()
    df["is_self_loop"] = (df["src"] == df["dest"]).astype("int8")
    df = add_subsplit(df)

    # vocab chốt trên train để tập cột ổn định giữa 3 file
    tr = df[df["split"] == "train"]
    formats    = sorted(tr["Payment Format"].unique())
    currencies = sorted(tr["Payment Currency"].unique())

    feats = {n: build_window(df, *w, formats, currencies) for n, w in WINDOWS.items()}
    cols = list(feats["train"].columns)
    feats = {n: f[cols] for n, f in feats.items()}          # khoá thứ tự cột

    verify(df, feats, formats, currencies)
    feats = rank_edges(feats, EDGE_RANK_COLS)

    for name, f in feats.items():
        p = EDGE_ATTR.format(name)
        f.reset_index().to_parquet(p, index=False)
        print(f"  đã lưu {p} shape={f.shape}")


if __name__ == "__main__":
    main()
