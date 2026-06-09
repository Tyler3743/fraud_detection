"""
Temporal split (60/20/20) cho HI-Small_Trans.csv  (node classification AML).

Nguyen tac:
- AML chia theo THOI GIAN (chronological), khong random -> tranh ro ri tuong lai.
- Cat theo ranh gioi DAU NGAY (00:00) vi he thong refresh node embedding moi ngay;
  cat giua ngay se gay leakage. Do do ty le thuc te lech nhe (54.5/19/26.5).
- So sanh theo chuoi ngay 'YYYY/MM/DD' (sort lexicographic = sort thoi gian)
  -> khong can parse datetime, chay nhanh tren 5 trieu dong.
- Giu cot bank/account dang string de khong mat so 0 dau (vd "001").
"""
import pandas as pd
import numpy as np

TRANS_PATH = "dataset_high/HI-Small_Trans.csv"
OUT_PATH = "dataset_high/HI-Small_Trans_split.csv"
TRAIN_FRAC, VAL_FRAC = 0.60, 0.80  # diem cat 60% va 80%

dtype = {"From Bank": str, "To Bank": str, "Account": str, "Account.1": str}
df = pd.read_csv(TRANS_PATH, dtype=dtype)

df["day"] = df["Timestamp"].str.slice(0, 10)
order = df["Timestamp"].sort_values(kind="mergesort").index
n = len(df)
days_sorted = df["day"].loc[order].to_numpy()
d1 = days_sorted[int(n * TRAIN_FRAC)]  # ngay chua moc 60% -> dau ngay nay = train|val
d2 = days_sorted[int(n * VAL_FRAC)]    # ngay chua moc 80% -> dau ngay nay = val|test

df["split"] = np.where(df["day"] < d1, "train",
                np.where(df["day"] < d2, "val", "test"))

print(f"Tong so giao dich: {n:,}")
print(f"Ranh gioi train|val = dau ngay {d1}")
print(f"Ranh gioi val|test  = dau ngay {d2}\n")
for s in ["train", "val", "test"]:
    sub = df[df.split == s]
    f = int((sub["Is Laundering"] == 1).sum())
    print(f"{s:5s} n={len(sub):>9,} ({len(sub)/n*100:5.2f}%)  "
          f"fraud={f:>5} rate={f/len(sub)*100:.4f}%  ngay {sub.day.min()}..{sub.day.max()}")

df.drop(columns=["day"]).to_csv(OUT_PATH, index=False)
print(f"\nDa luu: {OUT_PATH}")
