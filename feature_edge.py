import pandas as pd, numpy as np
from feature_node import load_data, scale_features   # tái dùng loader + scaler đã đúng

# Cột đếm/tiền lệch phải, không âm -> log1p. Vẫn nên profile skew bằng scaler.ipynb trước khi chốt.
EDGE_LOG1P_COLS = [
    "num_tx", "total_paid",
    "mean_paid", "std_paid", "min_paid", "max_paid",
    "active_day", "tx_per_day",
]

def compute_edge_feature(df_window: pd.DataFrame):
    df = df_window.copy()
    df["is_cross_bank"]     = (df["From Bank"] != df["To Bank"]).astype(int)
    df["is_cross_currency"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
    df["is_round_amount"]   = (df["Amount Paid"] % 1000 == 0).astype(int)
    df["is_self_loop"]      = (df["src"] == df["dest"]).astype(int)

    edge = df.groupby(["src", "dest"]).agg(
        num_tx          = ("Amount Paid", "size"),
        total_paid      = ("Amount Paid", "sum"),
        mean_paid       = ("Amount Paid", "mean"),
        std_paid        = ("Amount Paid", "std"),
        min_paid        = ("Amount Paid", "min"),
        max_paid        = ("Amount Paid", "max"),
        cross_ccy_ratio = ("is_cross_currency", "mean"),
        round_ratio     = ("is_round_amount", "mean"),
        is_cross_bank   = ("is_cross_bank", "max"),   # cố định trong 1 cặp
        is_self_loop    = ("is_self_loop", "max"),
        time_min        = ("Timestamp", "min"),
        time_max        = ("Timestamp", "max"),
        is_edge_mule    = ("Is Laundering", "max"),   # nhãn cạnh gộp
    )

    edge["active_day"] = (edge["time_max"] - edge["time_min"]).dt.days.clip(lower=1)
    edge["tx_per_day"] = edge["num_tx"] / edge["active_day"]
    edge = edge.drop(columns=["time_min", "time_max"]).fillna(0)  # std cạnh 1-tx -> 0

    final_cols = [
        "num_tx", "total_paid",
        "mean_paid", "std_paid", "min_paid", "max_paid",
         "cross_ccy_ratio",
        "round_ratio",  
        "is_cross_bank", "is_self_loop",
        "active_day", "tx_per_day",
        "is_edge_mule",
    ]
    return edge[final_cols]

def edge_payment_format_proportion(edge_feature, df_window, formats=None):
    if formats is None:
        formats = sorted(df_window["Payment Format"].unique())
    dummies = pd.get_dummies(df_window["Payment Format"]).astype(float)
    dummies = dummies.reindex(columns=formats, fill_value=0)
    prop = dummies.groupby([df_window["src"], df_window["dest"]]).mean()
    prop.columns = [f"pf_{c.replace(' ', '_')}" for c in formats]   # tỉ lệ [0,1] -> KHÔNG log1p
    res = edge_feature.join(prop)
    res[prop.columns] = res[prop.columns].fillna(0)
    return res, formats

def edge_currency_proportion(edge_feature, df_window, currencies=None,
                             col="Payment Currency", prefix="ccy"):
    if currencies is None:
        currencies = sorted(df_window[col].unique())
    dummies = pd.get_dummies(df_window[col]).astype(float)
    dummies = dummies.reindex(columns=currencies, fill_value=0)
    prop = dummies.groupby([df_window["src"], df_window["dest"]]).mean()
    prop.columns = [f"{prefix}_{c.replace(' ', '_')}" for c in currencies]
    res = edge_feature.join(prop)
    res[prop.columns] = res[prop.columns].fillna(0)
    return res, currencies

def main():
    df = load_data()
    masks = {                                   # cửa sổ LŨY TIẾN, giống node
        "train": df["split"] == "train",
        "val":   df["split"].isin(["train", "val"]),
        "test":  df["split"].notna(),
    }
    feats, labels, formats, currencies = {}, {}, None, None
    for name in ["train", "val", "test"]:
        window = df[masks[name]]
        ef = compute_edge_feature(window)
        ef, formats    = edge_payment_format_proportion(ef, window, formats)
        ef, currencies = edge_currency_proportion(ef, window, currencies)   # <-- thêm dòng này
        labels[name] = ef["is_edge_mule"]
        feats[name]  = ef.drop(columns=["is_edge_mule"])

    feats["train"], feats["val"], feats["test"], scaler = scale_features(
        feats["train"], feats["val"], feats["test"], log1p_cols=EDGE_LOG1P_COLS
    )
    for name in ["train", "val", "test"]:
        feats[name]["is_edge_mule"] = labels[name]         # gắn nhãn lại SAU scale
        p = f"dataset_high/edge_features_{name}.csv"
        feats[name].to_csv(p, index=True, index_label=["src", "dest"])
        n = int(feats[name]["is_edge_mule"].sum())
        print(f"  {name:5s}: {len(feats[name]):>8,} edges | {n:>5,} mule "
              f"({n/len(feats[name])*100:.3f}%)")

if __name__ == "__main__":
    main()