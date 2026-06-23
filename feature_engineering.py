import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load CSV, parse timestamp, tạo node ID src/dest/Is_Mule."""
    dtype_dict = {"From Bank": str, "Account": str, "To Bank": str, "Account.1": str}
    df = pd.read_csv(path, dtype=dtype_dict)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["src"]     = df["From Bank"] + "|" + df["Account"]
    df["dest"]    = df["To Bank"]   + "|" + df["Account.1"]
    df["Is_Mule"] = df["Is Laundering"] == 1
    return df


# ─────────────────────────────────────────────
# 2. TÍNH NODE FEATURE
# ─────────────────────────────────────────────

def compute_features(df_split: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 20 node feature từ một split (train / val / test).
    Trả về DataFrame index = node_id.
    """

    # --- Sender (groupby src) ---
    sender = df_split.groupby("src").agg(
        num_gd_gui      = ("Amount Paid",        "size"),
        tong_chi        = ("Amount Paid",        "sum"),
        mean_chi        = ("Amount Paid",        "mean"),
        std_chi         = ("Amount Paid",        "std"),
        out_degree      = ("dest",               "nunique"),
        num_bank_out    = ("To Bank",            "nunique"),
        num_currency_out= ("Payment Currency",   "nunique"),
        num_round_out   = ("Amount Paid",        lambda x: (x == x.astype(int)).sum()),
        timestamp_min   = ("Timestamp",          "min"),
        timestamp_max   = ("Timestamp",          "max"),
    )

    # --- Receiver (groupby dest) ---
    receiver = df_split.groupby("dest").agg(
        num_gd_nhan      = ("Amount Received",    "size"),
        tong_nhan        = ("Amount Received",    "sum"),
        mean_nhan        = ("Amount Received",    "mean"),
        std_nhan         = ("Amount Received",    "std"),
        in_degree        = ("src",                "nunique"),
        num_bank_in      = ("From Bank",          "nunique"),
        num_currency_in  = ("Receiving Currency", "nunique"),
        num_round_in     = ("Amount Received",    lambda x: (x == x.astype(int)).sum()),
    )

    # --- Merge ---
    feat = sender.join(receiver, how="outer")

    # --- Fill NaN (node chỉ gửi hoặc chỉ nhận) ---
    fill_zero = ["tong_nhan", "std_nhan", "num_gd_nhan", "in_degree",
                 "num_bank_in", "num_currency_in", "num_round_in"]
    feat[fill_zero] = feat[fill_zero].fillna(0)
    feat["mean_nhan"] = feat["mean_nhan"].fillna(feat["mean_nhan"].median())

    # --- Derived features ---
    feat["net_flow"]  = feat["tong_nhan"] - feat["tong_chi"]
    feat["num_days"]  = (feat["timestamp_max"] - feat["timestamp_min"]).dt.days + 1
    feat["gd_per_day"]= (feat["num_gd_gui"] / feat["num_days"]).fillna(feat["num_gd_gui"])

    feat["tile_cross_bank"] = (
        (feat["num_bank_out"] > 1) | (feat["num_bank_in"] > 1)
    ).astype(float)

    feat["tile_currency_mismatch"] = (
        df_split.groupby("src")
                .apply(lambda x: (x["Receiving Currency"] != x["Payment Currency"]).mean())
                .reindex(feat.index)
                .fillna(0)
    )

    feat["tile_round_amount"] = (
        (feat["num_round_out"] + feat["num_round_in"]) /
        (feat["num_gd_gui"]   + feat["num_gd_nhan"])
    ).fillna(0)

    # --- Label ---
    mule_nodes = set(df_split.loc[df_split["Is_Mule"], "src"]) | \
                 set(df_split.loc[df_split["Is_Mule"], "dest"])
    feat["Is_mule"] = feat.index.isin(mule_nodes).astype(int)

    # --- Select final columns ---
    final_cols = [
        "tong_chi", "tong_nhan", "mean_chi", "mean_nhan", "std_chi", "std_nhan", "net_flow",
        "out_degree", "in_degree",
        "num_bank_out", "num_bank_in", "tile_cross_bank",
        "num_currency_out", "num_currency_in", "tile_currency_mismatch",
        "num_gd_gui", "num_gd_nhan", "num_days", "gd_per_day",
        "tile_round_amount",
        "Is_mule",
    ]
    return feat[final_cols]


# ─────────────────────────────────────────────
# 3. TARGET ENCODING
# ─────────────────────────────────────────────

def add_target_encoding(
    df: pd.DataFrame,
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Thêm cột payment_format_encoded vào train / val / test.

    Cách hoạt động:
    - Fit encoding map trên train: Payment Format → mean(Is Laundering)
    - Apply map sang val và test (không fit lại → tránh leakage)
    - Node không có Payment Format → dùng global mean
    """
    df_train = df[df["split"] == "train"]
    global_mean     = df_train["Is Laundering"].mean()
    encoding_map    = df_train.groupby("Payment Format")["Is Laundering"].mean()

    for features, split_name in [
        (features_train, "train"),
        (features_val,   "val"),
        (features_test,  "test"),
    ]:
        df_split = df[df["split"] == split_name]

        # Lấy payment format phổ biến nhất của mỗi node (src)
        node_format = (
            df_split.groupby("src")["Payment Format"]
                    .agg(lambda x: x.mode()[0])
        )

        features["payment_format_encoded"] = (
            features.index.map(node_format)
                          .map(encoding_map)
                          .fillna(global_mean)
        )

    return features_train, features_val, features_test


# ─────────────────────────────────────────────
# 4. SCALE
# ─────────────────────────────────────────────

def scale_features(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Log1p transform + StandardScaler.
    Fit trên train, transform val và test (tránh leakage).
    Giữ nguyên cột Is_mule (không scale nhãn).
    """
    numeric_cols = [c for c in features_train.columns if c != "Is_mule"]

    scaler = StandardScaler()
    X_train = np.log1p(features_train[numeric_cols])
    X_val   = np.log1p(features_val[numeric_cols])
    X_test  = np.log1p(features_test[numeric_cols])

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=numeric_cols, index=features_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=numeric_cols, index=features_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=numeric_cols, index=features_test.index
    )

    # Gắn lại nhãn
    for scaled, original in [
        (X_train_scaled, features_train),
        (X_val_scaled,   features_val),
        (X_test_scaled,  features_test),
    ]:
        scaled["Is_mule"] = original["Is_mule"].values

    return X_train_scaled, X_val_scaled, X_test_scaled


# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────

def save_features(
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
    out_dir: str = "dataset_high",
) -> None:
    """Xuất 3 file CSV ra disk."""
    features_train.to_csv(f"{out_dir}/node_features_train.csv")
    features_val.to_csv(f"{out_dir}/node_features_val.csv")
    features_test.to_csv(f"{out_dir}/node_features_test.csv")
    print(f"✓ Xuất train ({len(features_train):,}) / val ({len(features_val):,}) / test ({len(features_test):,})")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING NODE (modular sample)")
    print("=" * 60)

    # Bước 1: Load
    df = load_data("dataset_high/HI-Small_Trans_split_index.csv")
    print(f"✓ Load {len(df):,} giao dịch")

    # Bước 2: Tính feature
    features_train = compute_features(df[df["split"] == "train"])
    features_val   = compute_features(df[df["split"] == "val"])
    features_test  = compute_features(df[df["split"] == "test"])
    print(f"✓ Nodes: train={len(features_train):,} val={len(features_val):,} test={len(features_test):,}")

    # Bước 3: Target encoding
    features_train, features_val, features_test = add_target_encoding(
        df, features_train, features_val, features_test
    )
    print("✓ Thêm payment_format_encoded")

    # Bước 4: Scale
    features_train, features_val, features_test = scale_features(
        features_train, features_val, features_test
    )
    print("✓ Log1p + StandardScaler")

    # Bước 5: Save
    save_features(features_train, features_val, features_test)

    # Thống kê
    for name, feat in [("train", features_train), ("val", features_val), ("test", features_test)]:
        n_mule = feat["Is_mule"].sum()
        print(f"  {name:5s}: {len(feat):>6,} nodes | {n_mule:>5,} mule ({n_mule/len(feat)*100:.2f}%)")


if __name__ == "__main__":
    main()
