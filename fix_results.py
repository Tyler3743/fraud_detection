import csv
import pandas as pd

rows = list(csv.reader(open("results.csv", encoding="utf-8")))
header = rows[0]                                   

header16 = header[:13] + ["train_time_s"] + header[13:]

fixed = []
for r in rows[1:]:
    if len(r) == 15:                
        r = r[:13] + [""] + r[13:]  
    fixed.append(r)                

df = pd.DataFrame(fixed, columns=header16)
df.to_csv("results.csv", index=False)
print(f"OK: {len(df)} dong, {len(df.columns)} cot")