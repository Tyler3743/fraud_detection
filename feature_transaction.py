import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SRC = "dataset_high/HI-Small_Trans_split_index.csv"
OUT = "dataset_high/transaction_features"

SCALE_COLS = ["amt_paid_log", "amt_recv_log", "amt_ratio_log", "hour", "day_of_week"]

def load_data():
    # Sử dụng engine pyarrow để đọc nhanh và nhẹ RAM
    dt = {"From Bank": str, "Account": str, "To Bank": str, "Account.1": str, "split": str}
    df = pd.read_csv(SRC, dtype=dt, engine="pyarrow")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["src"]  = df["From Bank"] + " | " + df["Account"]
    df["dest"] = df["To Bank"]   + " | " + df["Account.1"]
    return df

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # --- Tiền: Dùng log1p để tránh log(0) ---
    out["amt_paid_log"] = np.log1p(df["Amount Paid"])
    out["amt_recv_log"] = np.log1p(df["Amount Received"])
    
    # Sửa lỗi chia cho 0 và log(0) tại đây
    ratio = df["Amount Received"] / (df["Amount Paid"] + 1e-8)
    out["amt_ratio_log"] = np.log1p(ratio)

    # --- Cơ nghiệp vụ ---
    out["is_cross_bank"]     = (df["From Bank"] != df["To Bank"]).astype("int8")
    out["is_cross_currency"] = (df["Receiving Currency"] != df["Payment Currency"]).astype("int8")
    out["is_round_1000"]     = (df["Amount Paid"] % 1000 == 0).astype("int8")
    out["is_round_100"]      = (df["Amount Paid"] % 100  == 0).astype("int8")
    out["is_self_loop"]      = (df["src"] == df["dest"]).astype("int8")

    # --- Thời gian ---
    out["hour"]        = df["Timestamp"].dt.hour.astype("int16")
    out["day_of_week"] = df["Timestamp"].dt.dayofweek.astype("int16")

    # --- One-hot (Tối ưu RAM bằng cách ép dtype ngay từ đầu) ---
    pf  = pd.get_dummies(df["Payment Format"],   prefix="pf", dtype="int8")
    ccy = pd.get_dummies(df["Payment Currency"], prefix="ccy", dtype="int8")
    out = pd.concat([out, pf, ccy], axis=1)

    # --- Khóa join và nhãn (Để kiểu string/object để ghi Parquet không lỗi) ---
    out["src"]           = df["src"]
    out["dest"]          = df["dest"]
    out["Is Laundering"] = df["Is Laundering"].astype("int8")
    out["split"]         = df["split"]
    return out

def scale_features(feat: pd.DataFrame, scale_cols=SCALE_COLS):
    """Fit StandardScaler CHỈ trên split=='train', transform toàn bộ để tránh data leakage."""
    tr = feat["split"] == "train"
    scaler = StandardScaler().fit(feat.loc[tr, scale_cols].values)
    
    out = feat.copy()
    out[scale_cols] = scaler.transform(feat[scale_cols].values)
    return out, scaler

def main():
    print("--- 1. Loading data ---")
    df = load_data()
    
    print("--- 2. Building features ---")
    feat = build_transaction_features(df)
    
    print("--- 3. Scaling features (Train-only fit) ---")
    feat, scaler = scale_features(feat)
    
    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs("dataset_high", exist_ok=True)

    print("--- 4. Saving features ---")
    try:
        path = OUT + ".parquet"
        feat.to_parquet(path, index=False)
        print(f"[Success] Đã lưu dạng Parquet: {path}")
    except Exception as e:
        path = OUT + ".csv"
        feat.to_csv(path, index=False)
        print(f"[Warn] Không lưu được Parquet ({e}). Đã fallback lưu CSV: {path}")

    # In thông tin kiểm tra data split
    n_feat = feat.shape[1] - 4  # trừ src, dest, Is Laundering, split
    print(f"\nThống kê: rows={len(feat):,} | n_features={n_feat}")
    
    g = feat.groupby("split", observed=True)["Is Laundering"]
    rep = pd.DataFrame({"n": g.size(), "n_pos": g.sum(), "pos_%": (g.mean() * 100).round(3)})
    print(rep)
    f = pd.read_parquet("dataset_high/transaction_features.parquet")
    print(f[f.split=="train"]["amt_paid_log"].mean())   

if __name__ == "__main__":
    main()