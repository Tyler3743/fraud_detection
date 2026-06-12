import pandas as pd
import numpy as np

TRANS_PATH = "dataset_high/HI-Small_Trans.csv"
OUT_PATH = "dataset_high/HI-Small_Trans_split_day.csv"
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

df=df.iloc[order].reset_index(drop=True)

df.to_csv(OUT_PATH, index=False)
print(f"\nDa luu: {OUT_PATH}")
