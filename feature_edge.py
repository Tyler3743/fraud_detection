import pandas as pd, numpy as np
from feature_node import load_data, scale_features   # tái dùng loader + scaler đã đúng

# Cột đếm/tiền lệch phải, không âm -> log1p. Vẫn nên profile skew bằng scaler.ipynb trước khi chốt.
EDGE_LOG1P_COLS = [
    "e_num_tx", "e_total_paid", "e_total_received",
    "e_mean_paid", "e_std_paid", "e_min_paid", "e_max_paid",
    "e_active_day", "e_tx_per_day",
]

def compute_edge_feature(df_window: pd.DataFrame):
    df = df_window.copy()
    df["is_cross_bank"]     = (df["From Bank"] != df["To Bank"]).astype(int)
    df["is_cross_currency"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
    df["is_round_amount"]   = (df["Amount Paid"] % 1000 == 0).astype(int)
    df["is_self_loop"]      = (df["src"] == df["dest"]).astype(int)
    # amt_ratio chỉ có nghĩa khi cùng tiền tệ; cross-currency thực chất là tỉ giá FX
    df["amt_ratio"]         = df["Amount Received"] / (df["Amount Paid"] + np.finfo(float).eps)

    edge = df.groupby(["src", "dest"]).agg(
        e_num_tx          = ("Amount Paid", "size"),
        e_total_paid      = ("Amount Paid", "sum"),
        e_total_received  = ("Amount Received", "sum"),
        e_mean_paid       = ("Amount Paid", "mean"),
        e_std_paid        = ("Amount Paid", "std"),
        e_min_paid        = ("Amount Paid", "min"),
        e_max_paid        = ("Amount Paid", "max"),
        e_n_pay_ccy       = ("Payment Currency", "nunique"),
        e_n_recv_ccy      = ("Receiving Currency", "nunique"),
        e_cross_ccy_ratio = ("is_cross_currency", "mean"),
        e_round_ratio     = ("is_round_amount", "mean"),
        e_amt_ratio_mean  = ("amt_ratio", "mean"),
        e_is_cross_bank   = ("is_cross_bank", "max"),   # cố định trong 1 cặp
        e_is_self_loop    = ("is_self_loop", "max"),
        e_time_min        = ("Timestamp", "min"),
        e_time_max        = ("Timestamp", "max"),
        e_is_edge_mule    = ("Is Laundering", "max"),   # nhãn cạnh gộp
    )

    edge["e_active_day"] = (edge["e_time_max"] - edge["e_time_min"]).dt.days.clip(lower=1)
    edge["e_tx_per_day"] = edge["e_num_tx"] / edge["e_active_day"]
    edge = edge.drop(columns=["e_time_min", "e_time_max"]).fillna(0)  # std cạnh 1-tx -> 0

    final_cols = [
        "e_num_tx", "e_total_paid", "e_total_received",
        "e_mean_paid", "e_std_paid", "e_min_paid", "e_max_paid",
        "e_n_pay_ccy", "e_n_recv_ccy", "e_cross_ccy_ratio",
        "e_round_ratio", "e_amt_ratio_mean",
        "e_is_cross_bank", "e_is_self_loop",
        "e_active_day", "e_tx_per_day",
        "e_is_edge_mule",
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

def main():
    df = load_data()
    masks = {                                   # cửa sổ LŨY TIẾN, giống node
        "train": df["split"] == "train",
        "val":   df["split"].isin(["train", "val"]),
        "test":  df["split"].notna(),
    }
    feats, labels, formats = {}, {}, None
    for name in ["train", "val", "test"]:
        window = df[masks[name]]
        ef = compute_edge_feature(window)
        ef, formats = edge_payment_format_proportion(ef, window, formats)
        labels[name] = ef["e_is_edge_mule"]                  # tách nhãn
        feats[name]  = ef.drop(columns=["e_is_edge_mule"])   # feature không chứa nhãn

    feats["train"], feats["val"], feats["test"], scaler = scale_features(
        feats["train"], feats["val"], feats["test"], log1p_cols=EDGE_LOG1P_COLS
    )
    for name in ["train", "val", "test"]:
        feats[name]["e_is_edge_mule"] = labels[name]         # gắn nhãn lại SAU scale
        p = f"dataset_high/edge_features_{name}.csv"
        feats[name].to_csv(p, index=True, index_label=["src", "dest"])
        n = int(feats[name]["e_is_edge_mule"].sum())
        print(f"  {name:5s}: {len(feats[name]):>8,} edges | {n:>5,} mule "
              f"({n/len(feats[name])*100:.3f}%)")

if __name__ == "__main__":
    main()