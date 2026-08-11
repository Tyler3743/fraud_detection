import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_node import load_data     

EDGE_LOG1P_COLS = [
    "num_tx", "total_paid", "mean_paid", "std_paid",
    "min_paid", "max_paid", "active_day", "tx_per_day",
]

# name -> (cửa sổ tính FEATURE = lagged, cửa sổ định nghĩa TẬP CẠNH = lũy tiến Altman)
WINDOWS = {
    "train": (["train"],        ["train"]),
    "val":   (["train"],        ["train", "val"]),
    "test":  (["train", "val"], ["train", "val", "test"]),
}


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
        # round_ratio     = ("is_round", "mean"),
        time_min        = ("Timestamp", "min"),
        time_max        = ("Timestamp", "max"),
    )
    agg["std_paid"] = agg["std_paid"].fillna(0.0)          
    agg["active_day"] = (agg["time_max"] - agg["time_min"]).dt.days.clip(lower=1)
    agg["tx_per_day"] = agg["num_tx"] / agg["active_day"]
    agg = agg.drop(columns=["time_min", "time_max"])

    keys = [win["src"], win["dest"]]
    for col, names, prefix in [("Payment Format", formats,    "pf"),#vòng lặp để tính one hot
                               ("Payment Currency", currencies, "ccy")]:
        d = pd.get_dummies(win[col]).astype("float32").reindex(columns=names, fill_value=0.0)
        prop = d.groupby(keys, sort=False).mean()
        prop.columns = [f"{prefix}_{c.replace(' ', '_')}" for c in names]
        agg = agg.join(prop)
    return agg


def build_window(df, src_splits, graph_splits, formats, currencies):
    s_window   = df[df["split"].isin(src_splits)]# thống kê đặc trưng, mô tả hành vi của cạnh
    g_window = df[df["split"].isin(graph_splits)]# chia lũy tiến
    edges = g_window.groupby(["src", "dest"], sort=False).agg( #thống kê cạnh theo từng cửa sổ lũy tiến
        is_cross_bank = ("is_cross_bank", "max"),
        is_self_loop  = ("is_self_loop",  "max"),
    ).astype("int8")

    agg = aggregate_edges(s_window, formats, currencies)# dataframe tổng hợp hành vi giao dịch giữa 2 người,

    feat = edges.join(agg, how="left")                     
    feat["seen_before"] = feat["num_tx"].notna().astype("int8")
    return feat.fillna(0.0)


def scale_edges(feats, log_cols):
    """log1p + StandardScaler fit CHỈ trên bảng 'train' (tính từ cửa sổ train)."""
    for name in feats:
        feats[name][log_cols] = np.log1p(feats[name][log_cols].clip(lower=0))
    scaler = StandardScaler().fit(feats["train"][log_cols].values)
    for name in feats:
        feats[name][log_cols] = scaler.transform(feats[name][log_cols].values)
    return feats, scaler


def verify(df, feats, formats, currencies):
    # 1. không có cột nào suy từ nhãn
    banned = {"is_edge_mule", "is_mule", "Is Laundering"}
    for name, f in feats.items():
        assert not (banned & set(f.columns)), f"{name} còn cột suy từ nhãn"

    # 2. edge_attr val chỉ phụ thuộc train: xáo Amount Paid trong val -> bảng không đổi
    shuffled = df.copy()
    m = shuffled["split"] == "val"
    rng = np.random.default_rng(0)
    shuffled.loc[m, "Amount Paid"] = rng.permutation(shuffled.loc[m, "Amount Paid"].values)
    ref = build_window(shuffled, *WINDOWS["val"], formats, currencies)
    base = build_window(df, *WINDOWS["val"], formats, currencies)
    assert ref.equals(base), "edge_attr val ĐANG phụ thuộc dữ liệu val -> leak"
    print("  PASS: edge_attr val bất biến khi xáo dữ liệu val")

    # 3. cold-start: tỉ lệ cạnh chưa từng thấy
    for name, f in feats.items():
        r = 1 - f["seen_before"].mean()
        print(f"  {name:5s}: {len(f):>9,} cạnh | cold-start {r*100:5.2f}%")


def main():
    df = load_data()
    df["is_self_loop"] = (df["src"] == df["dest"]).astype("int8") 
    # vocab chốt trên train để tập cột ổn định giữa 3 file
    tr = df[df["split"] == "train"]
    formats    = sorted(tr["Payment Format"].unique())
    currencies = sorted(tr["Payment Currency"].unique())

    feats = {n: build_window(df, *w, formats, currencies) for n, w in WINDOWS.items()}
    cols = list(feats["train"].columns)
    feats = {n: f[cols] for n, f in feats.items()}          # khoá thứ tự cột

    verify(df, feats, formats, currencies)
    feats, _ = scale_edges(feats, EDGE_LOG1P_COLS)

    for name, f in feats.items():
        p = f"dataset_high/edge_attr_{name}.parquet"
        f.reset_index().to_parquet(p, index=False)
        print(f"  đã lưu {p} shape={f.shape}")


if __name__ == "__main__":
    main()