"""
Feature Engineering Node - Hoàn chỉnh cho bước 4
Tính ~21 feature từ trans data, log1p+scale, fit train→transform val/test
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("FEATURE ENGINEERING NODE")
print("=" * 70)

# ===== LOAD DỮ LIỆU =====
print("\n[1] Load dữ liệu...")
dtype_dict = {"From Bank": str, "Account": str, "To Bank": str, "Account.1": str}
df = pd.read_csv("dataset_high/HI-Small_Trans_split_index.csv", dtype=dtype_dict)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Node ID = Bank | Account (tuple)
df["src"] = df["From Bank"] + "|" + df["Account"]
df["dest"] = df["To Bank"] + "|" + df["Account.1"]
df["Is_Mule"] = df["Is Laundering"] == 1

print(f"  Total: {len(df):,} giao dịch")
for s in ["train", "val", "test"]:
    n = len(df[df["split"] == s])
    print(f"    {s}: {n:,}")


# ===== FUNCTION TÍNH FEATURE =====
def compute_features(df_split):
    """Tính feature từ một subset (train/val/test)"""

    # === SENDER FEATURES (groupby src) ===
    sender = df_split.groupby("src").agg(
        num_gd_gui=("Amount Paid", "size"),                # Số GD gửi
        tong_chi=("Amount Paid", "sum"),                   # Tổng tiền gửi
        mean_chi=("Amount Paid", "mean"),                  # Trung bình tiền gửi
        std_chi=("Amount Paid", "std"),                    # Std tiền gửi
        out_degree=("dest", "nunique"),                    # Số người nhận khác nhau
        num_bank_out=("To Bank", "nunique"),               # Số bank nhận
        num_currency_out=("Payment Currency", "nunique"),  # Số loại currency
        num_round_amount=("Amount Paid", lambda x: (x == x.astype(int)).sum()),  # Số tiền tròn
        timestamp_min=("Timestamp", "min"),                # Ngày hoạt động đầu
        timestamp_max=("Timestamp", "max"),                # Ngày hoạt động cuối
    )

    # === RECEIVER FEATURES (groupby dest) ===
    receiver = df_split.groupby("dest").agg(
        num_gd_nhan=("Amount Received", "size"),           # Số GD nhận
        tong_nhan=("Amount Received", "sum"),              # Tổng tiền nhận
        mean_nhan=("Amount Received", "mean"),             # Trung bình tiền nhận
        std_nhan=("Amount Received", "std"),               # Std tiền nhận
        in_degree=("src", "nunique"),                      # Số người gửi khác nhau
        num_bank_in=("From Bank", "nunique"),              # Số bank gửi
        num_currency_in=("Receiving Currency", "nunique"), # Số loại currency nhận
        num_round_amount_recv=("Amount Received", lambda x: (x == x.astype(int)).sum()),
    )

    # === MERGE sender + receiver ===
    features = sender.join(receiver, how="outer")

    # === FILL NaN ===
    features["tong_nhan"] = features["tong_nhan"].fillna(0)
    features["mean_nhan"] = features["mean_nhan"].fillna(features["mean_nhan"].median())
    features["std_nhan"] = features["std_nhan"].fillna(0)
    features["num_gd_nhan"] = features["num_gd_nhan"].fillna(0)
    features["in_degree"] = features["in_degree"].fillna(0)
    features["num_bank_in"] = features["num_bank_in"].fillna(0)
    features["num_currency_in"] = features["num_currency_in"].fillna(0)
    features["num_round_amount_recv"] = features["num_round_amount_recv"].fillna(0)

    # === TÍNH DERIVED FEATURES ===
    features["net_flow"] = features["tong_nhan"] - features["tong_chi"]

    # Velocity
    features["num_days"] = (features["timestamp_max"] - features["timestamp_min"]).dt.days + 1
    features["gd_per_day"] = features["num_gd_gui"] / features["num_days"]
    features["gd_per_day"] = features["gd_per_day"].fillna(features["num_gd_gui"])

    # Cross-bank tỷ lệ
    features["tile_cross_bank"] = ((features["num_bank_out"] > 1) | (features["num_bank_in"] > 1)).astype(float)

    # Currency mismatch tỷ lệ
    currency_mismatch = df_split.groupby("src").apply(
        lambda x: (x["Receiving Currency"] != x["Payment Currency"]).mean()
    )
    features["tile_currency_mismatch"] = currency_mismatch

    # Round amount tỷ lệ
    features["tile_round_amount"] = (
        (features["num_round_amount"] + features["num_round_amount_recv"]) /
        (features["num_gd_gui"] + features["num_gd_nhan"])
    ).fillna(0)

    # === NHÃN ===
    laund = df_split[df_split["Is_Mule"]]
    set_mule = set(laund["src"]) | set(laund["dest"])
    features["Is_mule"] = features.index.isin(set_mule).astype(int)

    # === SELECT CÁC CỘT FINAL ===
    final_cols = [
        # Dòng tiền
        "tong_chi", "tong_nhan", "mean_chi", "mean_nhan", "std_chi", "std_nhan", "net_flow",
        # Đối tác
        "out_degree", "in_degree",
        # Xuyên ngân hàng
        "num_bank_out", "num_bank_in", "tile_cross_bank",
        # Tiền tệ
        "num_currency_out", "num_currency_in", "tile_currency_mismatch",
        # Velocity
        "num_gd_gui", "num_gd_nhan", "num_days", "gd_per_day",
        # Số tròn
        "tile_round_amount",
        # Nhãn
        "Is_mule",
    ]
    features = features[final_cols]

    return features


# ===== TÍNH FEATURE CHO TỪNG SPLIT =====
print("\n[2] Tính feature...")

df_train = df[df["split"] == "train"]
df_val = df[df["split"] == "val"]
df_test = df[df["split"] == "test"]

features_train = compute_features(df_train)
features_val = compute_features(df_val)
features_test = compute_features(df_test)

print(f"  Train: {len(features_train):,} nodes")
print(f"  Val:   {len(features_val):,} nodes")
print(f"  Test:  {len(features_test):,} nodes")


# ===== LOG1P TRANSFORM + SCALE =====
print("\n[3] Log1p + StandardScaler (fit train)...")

# Numeric columns (bỏ Is_mule)
numeric_cols = [col for col in features_train.columns if col != "Is_mule"]

# Log1p transform
X_train_log = np.log1p(features_train[numeric_cols])
X_val_log = np.log1p(features_val[numeric_cols])
X_test_log = np.log1p(features_test[numeric_cols])

# Fit scaler trên train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_val_scaled = scaler.transform(X_val_log)
X_test_scaled = scaler.transform(X_test_log)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=features_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=numeric_cols, index=features_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=features_test.index)

# Thêm nhãn
X_train_scaled["Is_mule"] = features_train["Is_mule"].values
X_val_scaled["Is_mule"] = features_val["Is_mule"].values
X_test_scaled["Is_mule"] = features_test["Is_mule"].values

print(f"  ✓ {len(numeric_cols)} numeric features scaled")


# ===== XUẤT FILE =====
print("\n[4] Xuất file...")

X_train_scaled.to_csv("dataset_high/node_features_train.csv")
X_val_scaled.to_csv("dataset_high/node_features_val.csv")
X_test_scaled.to_csv("dataset_high/node_features_test.csv")

print(f"  ✓ node_features_train.csv ({len(X_train_scaled):,} × {len(X_train_scaled.columns)})")
print(f"  ✓ node_features_val.csv ({len(X_val_scaled):,} × {len(X_val_scaled.columns)})")
print(f"  ✓ node_features_test.csv ({len(X_test_scaled):,} × {len(X_test_scaled.columns)})")


# ===== THỐNG KÊ =====
print("\n[5] Thống kê:")
for split, feat in [("train", X_train_scaled), ("val", X_val_scaled), ("test", X_test_scaled)]:
    n_mule = (feat["Is_mule"] == 1).sum()
    pct = n_mule / len(feat) * 100
    print(f"  {split:5s}: {len(feat):>6,} nodes | {n_mule:>5,} mule ({pct:>5.2f}%)")

print("\n" + "=" * 70)
print("✓ Hoàn thành!")
print("=" * 70)
