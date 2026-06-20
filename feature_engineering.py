"""
Feature Engineering for Node Classification (Tầng 1 GNN)
- Tính ~25 feature cho mỗi node (tài khoản)
- Fit scaler + log1p transform trên train → transform test
- Chống leakage: fit/transform riêng biệt train vs test
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load dữ liệu và chia train/val/test"""

    # Dtype string để giữ leading zero
    dtype_dict = {
        "From Bank": str, "Account": str,
        "To Bank": str, "Account.1": str
    }

    df = pd.read_csv("dataset_high/HI-Small_Trans_split_index.csv", dtype=dtype_dict)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Tạo node ID (tuple Bank + Account)
    df["src"] = df["From Bank"] + "|" + df["Account"]
    df["dest"] = df["To Bank"] + "|" + df["Account.1"]
    df["Is_Mule"] = df["Is Laundering"] == 1

    print(f"✓ Load {len(df):,} giao dịch")
    print(f"  - Train: {len(df[df['split']=='train']):,} giao dịch")
    print(f"  - Val:   {len(df[df['split']=='val']):,} giao dịch")
    print(f"  - Test:  {len(df[df['split']=='test']):,} giao dịch")

    return df


def compute_node_features_single(df_subset, target_col="Is Laundering"):
    """
    Tính feature cho một subset (train/val/test)

    Feature tính toán (~25 chiều):

    Dòng tiền (6):
    - tong_chi, tong_nhan, mean_chi, mean_nhan, std_chi, std_nhan
    - net_flow

    Đa dạng đối tác (2):
    - out_degree (số người nhận riêng biệt)
    - in_degree (số người gửi riêng biệt)

    Xuyên ngân hàng (2):
    - num_bank_out (số bank nhận)
    - tile_cross_bank (% GD xuyên bank)

    Tiền tệ (2):
    - num_currency (số loại tiền tệ)
    - tile_currency_mismatch (% GD tiền tệ khác)

    Hình thức thanh toán (1 target-encoded):
    - payment_format_encoded (mean encoding của target)

    Velocity (5):
    - num_days (số ngày hoạt động)
    - gd_per_day (trung bình GD/ngày)
    - time_diff_mean (khoảng cách trung bình giữa GD)
    - time_diff_std
    - time_diff_max

    Số tròn (1):
    - tile_round_amount (% GD số tiền tròn)

    Nhãn (1):
    - Is_mule
    """

    # === TÍNH FEATURE SENDER (groupby src) ===

    sender_data = df_subset.groupby("src").agg(
        # Dòng tiền
        tong_chi=("Amount Paid", "sum"),
        mean_chi=("Amount Paid", "mean"),
        std_chi=("Amount Paid", "std"),

        # Đa dạng đối tác
        out_degree=("dest", "nunique"),  # số người nhận riêng biệt

        # Xuyên ngân hàng
        num_bank_out=("To Bank", "nunique"),  # số bank nhận tới

        # Tiền tệ
        num_currency_out=("Payment Currency", "nunique"),

        # Hình thức thanh toán
        payment_format_mode=("Payment Format", lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]),

        # Velocity
        num_txn_out=("Amount Paid", "size"),  # số GD gửi (có dùng sau)
        timestamp_min=("Timestamp", "min"),
        timestamp_max=("Timestamp", "max"),

        # Số tròn
        num_round_amount=("Amount Paid", lambda x: (x == x.astype(int)).sum()),  # GD số tiền tròn

        # Nhãn
        Is_mule=("Is_Mule", "max"),  # node là mule nếu xuất hiện >=1 GD fraud
    ).rename(columns={
        "num_txn_out": "num_gd_gui"
    })

    # === TÍNH FEATURE RECEIVER (groupby dest) ===

    receiver_data = df_subset.groupby("dest").agg(
        # Dòng tiền
        tong_nhan=("Amount Received", "sum"),
        mean_nhan=("Amount Received", "mean"),
        std_nhan=("Amount Received", "std"),

        # Đa dạng đối tác
        in_degree=("src", "nunique"),  # số người gửi riêng biệt

        # Xuyên ngân hàng
        num_bank_in=("From Bank", "nunique"),  # số bank gửi từ

        # Tiền tệ
        num_currency_in=("Receiving Currency", "nunique"),

        # Nhãn (lấy max vì node có thể xuất hiện ở cả sender và receiver)
        Is_mule_recv=("Is_Mule", "max"),

        # Số tròn (receiver)
        num_round_amount_recv=("Amount Received", lambda x: (x == x.astype(int)).sum()),
        num_gd_nhan=("Amount Received", "size"),
    )

    # === MERGE sender + receiver feature ===

    features = sender_data.copy()

    # Join receiver data (outer merge → node chỉ gửi hoặc chỉ nhận cũng có feature)
    features = features.join(receiver_data, how="outer")

    # === POST-PROCESSING ===

    # Fill NaN cho node chỉ gửi hoặc chỉ nhận
    features["tong_nhan"] = features["tong_nhan"].fillna(0)
    features["mean_nhan"] = features["mean_nhan"].fillna(features["mean_nhan"].median())
    features["std_nhan"] = features["std_nhan"].fillna(0)
    features["in_degree"] = features["in_degree"].fillna(0)
    features["num_bank_in"] = features["num_bank_in"].fillna(0)
    features["num_currency_in"] = features["num_currency_in"].fillna(0)
    features["num_gd_nhan"] = features["num_gd_nhan"].fillna(0)
    features["num_round_amount_recv"] = features["num_round_amount_recv"].fillna(0)

    features["Is_mule_recv"] = features["Is_mule_recv"].fillna(0)
    features["Is_mule"] = (features["Is_mule"] | features["Is_mule_recv"]).astype(int)

    # Net flow
    features["net_flow"] = features["tong_nhan"] - features["tong_chi"]

    # Velocity: ngày hoạt động
    features["num_days"] = (features["timestamp_max"] - features["timestamp_min"]).dt.days + 1
    features["gd_per_day"] = features["num_gd_gui"] / features["num_days"]
    features["gd_per_day"] = features["gd_per_day"].fillna(features["num_gd_gui"])  # node hoạt động 1 ngày

    # Xuyên ngân hàng
    features["tile_cross_bank_out"] = (features["num_bank_out"] > 1).astype(float)
    features["tile_cross_bank_in"] = (features["num_bank_in"] > 1).astype(float)

    # Tiền tệ
    features["tile_currency_mismatch"] = df_subset.groupby("src").apply(
        lambda x: (x["Receiving Currency"] != x["Payment Currency"]).mean()
    ).reindex(features.index).fillna(0)

    # Số tròn
    features["tile_round_amount"] = (
        (features["num_round_amount"] + features["num_round_amount_recv"]) /
        (features["num_gd_gui"] + features["num_gd_nhan"])
    ).fillna(0)

    # Số tròn NaN
    features["tile_round_amount"] = features["tile_round_amount"].fillna(0)

    # === SELECT CÁC FEATURE CẦN ===

    feature_cols = [
        # Dòng tiền
        "tong_chi", "tong_nhan", "mean_chi", "mean_nhan",
        "std_chi", "std_nhan", "net_flow",
        # Đa dạng đối tác
        "out_degree", "in_degree",
        # Xuyên ngân hàng
        "num_bank_out", "num_bank_in", "tile_cross_bank_out", "tile_cross_bank_in",
        # Tiền tệ
        "num_currency_out", "num_currency_in", "tile_currency_mismatch",
        # Velocity
        "num_gd_gui", "num_gd_nhan", "num_days", "gd_per_day",
        # Số tròn
        "tile_round_amount",
        # Nhãn
        "Is_mule",
    ]

    features = features[feature_cols]

    print(f"✓ Tính {len(features):,} nodes, {len(feature_cols)} feature")

    return features


def apply_log1p_and_scale(X_train, X_test=None, scaler_type="standard"):
    """
    Log1p transform + StandardScaler/RobustScaler

    Args:
        X_train: DataFrame train features (fit scaler trên này)
        X_test: DataFrame test features (transform chỉ)
        scaler_type: "standard" (StandardScaler) hoặc "robust" (RobustScaler)

    Returns:
        X_train_scaled, X_test_scaled (hoặc None), scaler
    """

    # Log1p transform
    X_train_log = np.log1p(X_train)
    if X_test is not None:
        X_test_log = np.log1p(X_test)

    # Fit scaler trên train
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_log)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test_log)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, None, scaler


def add_target_encoding(df_trans, features_train, features_val, features_test, target_col="Is Laundering"):
    """
    Thêm target encoding cho Payment Format vào feature set

    Fit trên train, apply trên val/test
    """

    # Fit target encoding trên train
    df_train = df_trans[df_trans["split"] == "train"]
    payment_encoding = df_train.groupby("Payment Format")[target_col].mean()
    global_mean = df_train[target_col].mean()

    print(f"✓ Target encoding Payment Format (global mean: {global_mean:.6f})")
    print(f"  Encoding map:")
    print(payment_encoding.sort_values(ascending=False))

    # Apply encoding
    for features, name in [(features_train, "train"), (features_val, "val"), (features_test, "test")]:
        df_split = df_trans[df_trans["split"] == name]

        # Map payment format → target mean
        payment_to_target = dict(df_split.groupby("src").apply(
            lambda x: x.iloc[0]["Payment Format"]  # lấy payment format từ transaction
        ))

        # Ngoại lệ: nếu src có nhiều payment format, lấy mode
        payment_to_target = df_split.groupby("src")["Payment Format"].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )

        features["payment_format_encoded"] = features.index.map(payment_to_target).map(payment_encoding).fillna(global_mean)

    print(f"✓ Thêm payment_format_encoded vào features")

    return features_train, features_val, features_test


def main():
    """Main pipeline"""

    print("=" * 70)
    print("FEATURE ENGINEERING FOR NODE CLASSIFICATION")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data()

    # Tính feature cho từng split
    print("\n[1/4] Tính feature train...")
    features_train = compute_node_features_single(df[df["split"] == "train"])

    print("\n[2/4] Tính feature val...")
    features_val = compute_node_features_single(df[df["split"] == "val"])

    print("\n[3/4] Tính feature test...")
    features_test = compute_node_features_single(df[df["split"] == "test"])

    # Log1p + Scale (fit trên train)
    print("\n[4/4] Log1p transform + Scale...")

    # Chọn numeric feature (bỏ Is_mule)
    numeric_cols = [col for col in features_train.columns if col != "Is_mule"]

    X_train = features_train[numeric_cols]
    X_val = features_val[numeric_cols]
    X_test = features_test[numeric_cols]

    X_train_scaled, X_val_scaled, scaler = apply_log1p_and_scale(X_train, X_val, scaler_type="standard")
    X_test_scaled, _, _ = apply_log1p_and_scale(X_train, X_test, scaler_type="standard")  # reuse scaler
    X_test_scaled = scaler.transform(np.log1p(X_test))
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)

    # Thêm nhãn Is_mule
    features_train_final = X_train_scaled.copy()
    features_train_final["Is_mule"] = features_train["Is_mule"]

    features_val_final = X_val_scaled.copy()
    features_val_final["Is_mule"] = features_val["Is_mule"]

    features_test_final = X_test_scaled.copy()
    features_test_final["Is_mule"] = features_test["Is_mule"]

    # Xuất file
    print("\n" + "=" * 70)
    print("XUẤT FILE")
    print("=" * 70)

    features_train_final.to_csv("dataset_high/node_features_train.csv")
    features_val_final.to_csv("dataset_high/node_features_val.csv")
    features_test_final.to_csv("dataset_high/node_features_test.csv")

    print(f"✓ node_features_train.csv ({len(features_train_final):,} nodes, {len(features_train_final.columns)} features)")
    print(f"✓ node_features_val.csv ({len(features_val_final):,} nodes)")
    print(f"✓ node_features_test.csv ({len(features_test_final):,} nodes)")

    # Thống kê
    print("\nThống kê:")
    for split, feat in [("train", features_train_final), ("val", features_val_final), ("test", features_test_final)]:
        n_mule = (feat["Is_mule"] == 1).sum()
        print(f"  {split:5s}: {len(feat):>6,} nodes | {n_mule:>5,} mule ({n_mule/len(feat)*100:>5.2f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
