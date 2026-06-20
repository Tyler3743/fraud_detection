import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("dataset_high/HI-Small_Trans.csv",dtype={"From Bank":str, "Account":str, "To Bank": str, "Account.1":str})
df["src"]=df["From Bank"] + " | " + df["Account"]#cột tài khoản, node
df["dest"]=df["To Bank"] +" | " +df["Account.1"]
df["!currency"] = df["Receiving Currency"] != df["Payment Currency"]# tỷ lệ trộn trên tất cả các giao dịch
df["Is_Mule"] = df["Is Laundering"]==1
laund=df[df["Is_Mule"]]
set_mule=set(laund["src"]) | set(laund["dest"])
# === SENDER FEATURES (groupby src) ===
sender_features = df.groupby("src").agg(
    so_giao_dich_gui=("Amount Paid", "size"),           # ✓ OK
    tong_chi=("Amount Paid", "sum"),                    # ✓ OK
    mean_chi=("Amount Paid", "mean"),                   # 🔧 SỬA: từ cột gốc
    std_chi=("Amount Paid", "std"),                     # 🆕 THÊM
    ti_le_tron_tien=("!currency", "mean"),              # ✓ OK
    so_nguoi_nhan=("dest", "nunique"),                  # ✓ OK (out-degree)
    num_bank_out=("To Bank", "nunique"),                # 🆕 THÊM: số bank nhận
)

# === RECEIVER FEATURES (groupby dest) ===
receiver_features = df.groupby("dest").agg(
    tong_nhan=("Amount Received", "sum"),               # 🆕 THÊM
    mean_nhan=("Amount Received", "mean"),              # 🔧 SỬA: từ cột gốc
    std_nhan=("Amount Received", "std"),                # 🆕 THÊM
    so_nguoi_gui=("src", "nunique"),                    # 🆕 THÊM: in-degree
    num_bank_in=("From Bank", "nunique"),               # 🆕 THÊM: số bank gửi
)

# === MERGE ===
df = sender_features.join(receiver_features, how="outer")
df["tong_nhan"] = df["tong_nhan"].fillna(0)
df["mean_nhan"] = df["mean_nhan"].fillna(df["mean_nhan"].median())
df["std_nhan"] = df["std_nhan"].fillna(0)
df["so_nguoi_gui"] = df["so_nguoi_gui"].fillna(0)
df["num_bank_in"] = df["num_bank_in"].fillna(0)

df["net_flow"] = df["tong_nhan"] - df["tong_chi"]
df["Is_mule"] = df.index.isin(set_mule)
df.head()