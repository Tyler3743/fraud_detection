import os
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

# Cột cần log1p — đề xuất từ scaler.ipynb (skew_raw > 1 và min >= 0).
# net_flow có giá trị âm -> KHÔNG log1p; 3 cột *_ratio đã nằm [0,1] -> bỏ qua.
LOG1P_COLS = [
    "num_send", "out_degree", "tx_per_day", "std_send", "mean_send", "tong_gui",
    "tong_nhan", "std_receive", "mean_receive", "num_bank_out", "num_bank_in",
    "in_degree", "num_receive", "active_day",
]

def load_data():
    df=pd.read_csv("dataset_high/HI-Small_Trans.csv",dtype={"From Bank":str, "Account":str, "To Bank": str, "Account.1":str})
    df["Timestamp"]=pd.to_datetime(df["Timestamp"])
    df["src"]=df["From Bank"]+ " | "+df["Account"]
    df["dest"]=df["To Bank"]+ " | "+df["Account.1"]
    return df 
def compute_feature(df_split:pd.DataFrame):
    df_split=df_split.copy()
    df_split["is_cross_bank"]=(df_split["From Bank"]!=df_split["To Bank"]).astype(int)
    df_split["is_currency_change"]=(df_split["Receiving Currency"]!=df_split["Payment Currency"]).astype(int)
    df_split["is_round_amount"]=(df_split["Amount Paid"]%1000==0).astype(int)
    sender=df_split.groupby("src").agg(
        tong_gui=("Amount Paid","sum"),
        num_send=("Amount Paid","size"),
        mean_send=("Amount Paid","mean"),
        std_send=("Amount Paid","std"),
        num_bank_out=("To Bank","nunique"),
        currency_mix_out=("Payment Currency","nunique"),
        out_degree=("dest","nunique"),
        time_min=("Timestamp","min"),
        time_max=("Timestamp","max"),
        cross_bank_ratio=("is_cross_bank","mean"),
        cross_currency_ratio=("is_currency_change","mean"),
        round_amount_ratio=("is_round_amount","mean")
    )
    receiver=df_split.groupby("dest").agg(
        tong_nhan=("Amount Received","sum"),
        num_receive=("Amount Received","size"),
        mean_receive=("Amount Received","mean"),
        std_receive=("Amount Received","std"),
        num_bank_in=("From Bank","nunique"),
        currency_mix_in=("Receiving Currency","nunique"),
        in_degree=("src","nunique")
    )
    node_feature=sender.join(receiver, how="outer")
    node_feature["net_flow"]=node_feature["tong_nhan"].fillna(0) - node_feature["tong_gui"].fillna(0)
    node_feature["active_day"]=(node_feature["time_max"]-node_feature["time_min"]).dt.days
    node_feature["active_day"]=node_feature["active_day"].fillna(0).clip(lower=1)
    node_feature["tx_per_day"]=(node_feature["num_send"].fillna(0)+node_feature["num_receive"].fillna(0))/node_feature["active_day"]
    node_feature=node_feature.fillna(0)
    final_cols=["tong_gui","num_send","mean_send","std_send","num_bank_out","currency_mix_out","out_degree","cross_bank_ratio","round_amount_ratio",
                "cross_currency_ratio",
                "tong_nhan","num_receive","mean_receive","std_receive","num_bank_in","currency_mix_in","in_degree",
                "net_flow","active_day","tx_per_day"]
    return node_feature[final_cols]


def add_payment_format_te(node_feature, df_window, te_map=None, global_mean=None):
    """Target encoding cho 'Payment Format' rồi gộp về node theo 2 chiều.

    - 'Payment Format' và 'Is Laundering' nằm ở CẤP GIAO DỊCH; ta đổi nó thành số
      bằng cách map mỗi format -> mean(Is Laundering). Map học CHỈ trên train.
    - Mỗi giao dịch nhận giá trị TE theo format của nó; rồi gộp về node bằng cách
      lấy trung bình theo giao dịch GỬI (pf_te_out) và theo giao dịch NHẬN (pf_te_in).
    - Node không gửi (hoặc không nhận) -> điền bằng global_mean (base rate train,
      giá trị trung tính), không điền 0.

    Tham số:
      node_feature : DataFrame node (output compute_feature), index = khóa node.
      df_window    : DataFrame giao dịch của cửa sổ (đã có cột 'src', 'dest').
      te_map       : Series format->mean(Is Laundering). None = học trên df_window (train).
      global_mean  : mean(Is Laundering) toàn train, dùng cho format lạ / node thiếu chiều.

    Trả về: (node_feature_kèm_2_cột, te_map, global_mean).
    Lưu ý chống leakage: với val/test PHẢI truyền lại te_map & global_mean của train.
    Vì map là trung bình theo format trên hàng triệu giao dịch và Payment Format ít
    loại (~7), phần đóng góp của 1 node vào mean là không đáng kể -> bỏ qua smoothing.
    """
    if te_map is None:
        global_mean = float(df_window["Is Laundering"].mean())
        te_map = df_window.groupby("Payment Format")["Is Laundering"].mean()

    tx = df_window[["src", "dest", "Payment Format"]].copy()
    tx["pf_te"] = tx["Payment Format"].map(te_map).fillna(global_mean)
    pf_out = tx.groupby("src")["pf_te"].mean().rename("pf_te_out")
    pf_in = tx.groupby("dest")["pf_te"].mean().rename("pf_te_in")

    out = node_feature.join(pf_out).join(pf_in)
    out[["pf_te_out", "pf_te_in"]] = out[["pf_te_out", "pf_te_in"]].fillna(global_mean)
    return out, te_map, global_mean


def scale_features(train_df, val_df, test_df, log1p_cols=LOG1P_COLS):
    """log1p cho cột lệch phải rồi StandardScaler (z-score).

    - log1p áp cho các cột trong log1p_cols (đều >= 0).
    - StandardScaler FIT CHỈ trên train, sau đó transform val/test bằng cùng scaler
      (fit cả 3 tập = leakage).
    - val/test được căn theo đúng thứ tự cột của train trước khi xử lý.

    Trả về: (train_scaled, val_scaled, test_scaled, scaler).
    """
    cols = list(train_df.columns)
    use_log = [c for c in log1p_cols if c in cols]

    frames = {
        "train": train_df.copy(),
        "val": list(pd.concat(["train_df","val_df"])),
        "test": list(pd.concat(["train_df","val_df","test_df"])),
    }
    for f in frames.values():
        f[use_log] = np.log1p(f[use_log])

    scaler = StandardScaler().fit(frames["train"].values)
    scaled = {
        k: pd.DataFrame(scaler.transform(f.values), index=f.index, columns=cols)
        for k, f in frames.items()
    }
    return scaled["train"], scaled["val"], scaled["test"], scaler


def save_node_features(train_df, val_df, test_df, out_dir="dataset_high"):
    """Ghi ra 3 file CSV; giữ index (khóa node) bằng nhãn cột 'node'."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, f in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = os.path.join(out_dir, f"node_features_{name}.csv")
        f.to_csv(p, index=True, index_label="node")
        paths[name] = p
    return paths





