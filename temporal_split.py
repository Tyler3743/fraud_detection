import pandas as pd
df_trans=pd.read_csv("dataset_high/HI-Small_trans.csv")
df_trans["Timestamp"]=pd.to_datetime(df_trans["Timestamp"])
df_trans=df_trans.sort_values("Timestamp").reset_index(drop=True)
nums_trans=len(df_trans)
endpoint_train = int(nums_trans*0.6)
endpoint_val =  int(nums_trans*0.8)
t1_raw = df_trans.loc[endpoint_train,"Timestamp"]
t2_raw = df_trans.loc[endpoint_val,"Timestamp"]
t1=t1_raw.normalize()
t2=t2_raw.normalize()
df_trans["split"]="train"
df_trans.loc[(df_trans["Timestamp"]>=t1)&(df_trans["Timestamp"]<t2),"split"]="val"
df_trans.loc[(df_trans["Timestamp"]>=t2),"split"]="test"
print(f"endpoint tập train: {t1}")
print(f"endpoint tập val {t2}")
print(df_trans["split"].value_counts())
#df_trans.to_csv("dataset_high/Hi-Small_Trans_split.csv",index=False)