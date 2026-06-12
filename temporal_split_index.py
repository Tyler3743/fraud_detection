"""
Temporal split (60/20/20) theo INDEX chinh xac -- bien the Altman et al. 2023.

Nguyen tac:
- Sap xep theo Timestamp tang dan (mergesort, stable), sau do tim hai moc Timestamp
  ts1 = Timestamp tai index t1 = int(n*0.6), ts2 = Timestamp tai index t2 = int(n*0.8).
- Phan cong theo gia tri Timestamp (khong theo so index tho):
    train = Timestamp < ts1
    val   = ts1 <= Timestamp < ts2
    test  = Timestamp >= ts2
  Moi giao dich co Timestamp == ts1 duoc day TOAN BO vao VAL (khong bi chia doi).
  Moi giao dich co Timestamp == ts2 duoc day TOAN BO vao TEST.
  He qua: ti le thuc te lech nhe khoi 60/20/20 do don ca nhom cung Timestamp ve mot
  phia de tranh leakage hai chieu (cung mot thoi diem bi xep vao hai tap khac nhau).
- Bien the nay KHONG snap theo dau ngay, nen diem cat co the roi GIUA MOT NGAY.
  CANH BAO: neu dung cho pipeline GNN co co che refresh embedding theo ngay,
  phai dong bo moc refresh voi moc ts1/ts2 nay (khong phai theo ngay lich).
  Bien the snapshot-ngay (temporal_split.py) phu hop hon voi kien truc hien tai.
- Giu cot bank/account dang string de khong mat so 0 dau (vd "001", "0010").
"""
import pandas as pd

TRANS_PATH = "dataset_high/HI-Small_Trans.csv"
OUT_PATH = "dataset_high/HI-Small_Trans_split_index.csv"

dtype = {"From Bank": str, "To Bank": str, "Account": str, "Account.1": str}
df = pd.read_csv(TRANS_PATH, dtype=dtype)

order = df["Timestamp"].sort_values(kind="mergesort").index
df = df.iloc[order].reset_index(drop=True)

n = len(df)
t1 = int(n * 0.6)
t2 = int(n * 0.8)

# Lay Timestamp tai diem cat raw de lam nguong phan cong
ts1 = df["Timestamp"].iloc[t1]
ts2 = df["Timestamp"].iloc[t2]

# Phan cong theo gia tri Timestamp: toan bo nhom cung ts1 vao val, cung ts2 vao test
df["split"] = "test"
df.loc[df["Timestamp"] < ts2, "split"] = "val"
df.loc[df["Timestamp"] < ts1, "split"] = "train"

# Assert: khong co Timestamp nao xuat hien o hai tap (strict ordering)
assert df[df.split == "train"]["Timestamp"].max() < df[df.split == "val"]["Timestamp"].min(), \
    "Leakage: Timestamp chong lan giua train va val"
assert df[df.split == "val"]["Timestamp"].max() < df[df.split == "test"]["Timestamp"].min(), \
    "Leakage: Timestamp chong lan giua val va test"

print(f"Tong so giao dich: {n:,}")
print(f"Diem cat raw: t1=index {t1} (ts1={ts1}), t2=index {t2} (ts2={ts2})")
print(f"Ranh gioi thuc te sau tie-breaking:")
print(f"  train max Timestamp = {df[df.split=='train']['Timestamp'].max()}")
print(f"  val   min Timestamp = {df[df.split=='val']['Timestamp'].min()}")
print(f"  val   max Timestamp = {df[df.split=='val']['Timestamp'].max()}")
print(f"  test  min Timestamp = {df[df.split=='test']['Timestamp'].min()}\n")

for s in ["train", "val", "test"]:
    sub = df[df.split == s]
    f = int((sub["Is Laundering"] == 1).sum())
    print(f"{s:5s} n={len(sub):>9,} ({len(sub)/n*100:5.2f}%)  "
          f"fraud={f:>5} rate={f/len(sub)*100:.4f}%  "
          f"ngay {sub['Timestamp'].min()[:10]}..{sub['Timestamp'].max()[:10]}")

df.to_csv(OUT_PATH, index=False)
print(f"\nDa luu: {OUT_PATH}")

# Assert leading zero: kiem tra noi dung thuc te co gia tri bat dau bang '0'
assert df["From Bank"].str.match(r"^0").any(), "Khong tim thay leading zero trong From Bank"
assert df["To Bank"].str.match(r"^0").any(), "Khong tim thay leading zero trong To Bank"
print(f"\nLeading zero OK: From Bank mau = '{df['From Bank'].iloc[0]}', To Bank mau = '{df['To Bank'].iloc[0]}'")
