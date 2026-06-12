import pandas as pd

#ép kiểu phòng trường hợp pandas đọc thành int và có số 0 đầu sẽ bỏ số 0 đầu sẽ làm sai dữ liệu khi tạo node
dtype={"From Bank":str, "To Bank": str, "Account":str, "Account.1": str}
df_trans=pd.read_csv("dataset_high/HI-Small_Trans.csv")

#sắp xếp dataframe theo timestamp
order = df_trans["Timestamp"].sort_values(kind="mergesort").index
df_trans=df_trans.iloc[order].reset_index(drop=True)

#chia dữ liệu theo train/test/val
n=len(df)
t1=int(n*0.6)
t2=int(n*0.8)
ts1=df_trans["Timestamp"].iloc[t1]
ts2=df_trans["Timestamp"].iloc[t2]

# gán nhãn train/test/value cho tập dữ liệu mới và xuất file
df_trans["split"]="test"
df_trans.loc[df_trans["Timestamp"]<ts2,"split"]="val"
df_trans.loc[df_trans["Timestamp"]<ts1,"split"]="train"
assert df[df.split == "train"]["Timestamp"].max() < df[df.split == "val"]["Timestamp"].min(), \
    "Leakage: Timestamp chong lan giua train va val"
assert df[df.split == "val"]["Timestamp"].max() < df[df.split == "test"]["Timestamp"].min(), \
    "Leakage: Timestamp chong lan giua val va test"

# xuất file csv
df_trans.to_csv("dataset_high/HI-Small_Trans_split_index.csv", index=False)
print(f"Đã lưu: {dataset_high/HI-Small_Trans_split_index.csv}")
