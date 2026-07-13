import os
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

# Cột cần log1p — đề xuất từ scaler.ipynb (skew_raw > 1 và min >= 0).
# net_flow có giá trị âm -> KHÔNG log1p; 3 cột *_ratio đã nằm [0,1] -> bỏ qua.
LOG1P_COLS = [
    "num_send", "out_degree", "tx_per_day", "std_send", "mean_send", "tong_gui",
    "tong_nhan", "std_receive", "mean_receive", "num_bank_out", "num_bank_in",
    "in_degree", "num_receive", "active_day","max_send","min_send","max_receive","min_receive","median_send","median_receive"
]

def load_data():
    df=pd.read_csv("dataset_high/HI-Small_Trans_split_index.csv",dtype={"From Bank":str, "Account":str, "To Bank": str, "Account.1":str})
    df["Timestamp"]=pd.to_datetime(df["Timestamp"])
    df["src"]=df["From Bank"]+ " | "+df["Account"]
    df["dest"]=df["To Bank"]+ " | "+df["Account.1"]
    return df 
def compute_feature(df_split:pd.DataFrame):
    df_split=df_split.copy()
    df_split["is_cross_bank"]=(df_split["From Bank"]!=df_split["To Bank"]).astype(int)
    df_split["is_currency_change"]=(df_split["Receiving Currency"]!=df_split["Payment Currency"]).astype(int)
    df_split["is_round_amount"]=(df_split["Amount Paid"]%1000==0).astype(int)
    sender=df_split.groupby("src").agg(
        tong_gui=("Amount Paid","sum"),
        num_send=("Amount Paid","size"),
        mean_send=("Amount Paid","mean"),
        std_send=("Amount Paid","std"),
        num_bank_out=("To Bank","nunique"),
        currency_mix_out=("Payment Currency","nunique"),
        out_degree=("dest","nunique"),
        time_min1=("Timestamp","min"),
        time_max1=("Timestamp","max"),
        cross_bank_ratio=("is_cross_bank","mean"),
        cross_currency_ratio=("is_currency_change","mean"),
        round_amount_ratio=("is_round_amount","mean"),
        median_send = ("Amount Paid", "median"),
        skew_send   = ("Amount Paid", "skew"),
        kurt_send   = ("Amount Paid", lambda s: s.kurt()),
        min_send=("Amount Paid", "min"),
        max_send=("Amount Paid", "max")
    )
    receiver=df_split.groupby("dest").agg(
        tong_nhan=("Amount Received","sum"),
        num_receive=("Amount Received","size"),
        mean_receive=("Amount Received","mean"),
        std_receive=("Amount Received","std"),
        num_bank_in=("From Bank","nunique"),
        currency_mix_in=("Receiving Currency","nunique"),
        in_degree=("src","nunique"),
        time_min2=("Timestamp","min"),
        time_max2=("Timestamp","max"),
        median_receive = ("Amount Received", "median"),
        skew_receive   = ("Amount Received", "skew"),
        kurt_receive   = ("Amount Received", lambda s: s.kurt()),
        min_receive=("Amount Received", "min"),
        max_receive=("Amount Received", "max")
    )
    node_feature=sender.join(receiver, how="outer")
    node_feature["net_flow"]=node_feature["tong_nhan"].fillna(0) - node_feature["tong_gui"].fillna(0)
    node_feature["time_max"]=node_feature[["time_max1","time_max2"]].max(axis=1)
    node_feature["time_min"]=node_feature[["time_min1","time_min2"]].min(axis=1)
    node_feature["active_day"]=(node_feature["time_max"]-node_feature["time_min"]).dt.days
    node_feature["active_day"]=node_feature["active_day"].fillna(0).clip(lower=1)
    node_feature["tx_per_day"]=(node_feature["num_send"].fillna(0)+node_feature["num_receive"].fillna(0))/node_feature["active_day"]
    node_feature=node_feature.fillna(0)
    node_feature["net_flow_ratio"]=node_feature["net_flow"].abs()/(node_feature["tong_gui"]+node_feature["tong_nhan"]+np.finfo(float).eps)
    mule_nodes=(
        set(df_split.loc[df_split["Is Laundering"]==1,"src"])|
        set(df_split.loc[df_split["Is Laundering"]==1,"dest"])
    )
    node_feature["is_mule"]=node_feature.index.isin(mule_nodes).astype(int)
    final_cols=["tong_gui","num_send","mean_send","std_send","num_bank_out","currency_mix_out","out_degree","cross_bank_ratio","round_amount_ratio",
                "cross_currency_ratio", "median_send", "skew_send", "kurt_send","median_receive","skew_receive","kurt_receive",
                "tong_nhan","num_receive","mean_receive","std_receive","num_bank_in","currency_mix_in","in_degree",
                "net_flow","active_day","tx_per_day","is_mule","net_flow_ratio","min_send","max_send","min_receive","max_receive"]
    
    return node_feature[final_cols]

def payment_format_proportion(node_feature, df_window, format=None):
    if format is None:
        format = sorted(df_window["Payment Format"].unique())
    tx= df_window[["src","dest","Payment Format"]].copy()
    dummies=pd.get_dummies(tx["Payment Format"]).astype(float)
    dummies=dummies.reindex(columns=format,fill_value=0)
    out_prop = dummies.groupby(tx["src"]).mean()
    out_prop.columns=[f"pf_{c.replace(' ','_')}_out" for c in format]
    in_prop = dummies.groupby(tx["dest"]).mean()
    in_prop.columns=[f"pf_{c.replace(' ','_')}_in" for c in format]

    res=node_feature.join(out_prop).join(in_prop)
    pf_cols=list(out_prop.columns)+list(in_prop.columns)
    res[pf_cols]=res[pf_cols].fillna(0)
    return res, format

def scale_features(train_df, val_df, test_df, log1p_cols=LOG1P_COLS):
    cols=list(train_df.columns)
    use_log=[c for c in log1p_cols if c in cols]
    frames = {
        "train": train_df.copy(),
        "val": val_df.reindex(columns=cols).copy(),
        "test": test_df.reindex(columns=cols).copy(),
    }
    for f in frames.values():
        f[use_log]=np.log1p(f[use_log])
    scaler=StandardScaler().fit(frames["train"].values)
    scaled={
        k:pd.DataFrame(scaler.transform(f.values),index=f.index,columns=cols)
        for k,f in frames.items()
    }
    return scaled["train"], scaled["val"], scaled["test"], scaler
def save_feature(train_df, val_df, test_df, out_dir="dataset_high"):
    os.makedirs(out_dir,exist_ok=True)
    paths={}
    for name, f in [("train",train_df),("val",val_df),("test",test_df)]:
        p = os.path.join(out_dir, f"node_features_{name}.csv")
        f.to_csv(p, index=True, index_label="node")
        paths[name] = p
    return paths
def main():
    df=load_data()
    masks={
        "train": df["split"]=="train",
        "val": df["split"].isin(["train","val"]),
        "test": df["split"].notna()
    }
    feats, labels, formats = {}, {}, None
    for name in ["train", "val", "test"]:
        window=df[masks[name]]
        nf=compute_feature(window)
        nf, formats=payment_format_proportion(nf,window,formats)
        labels[name] = nf["is_mule"]                   # tách label ra
        feats[name] = nf.drop(columns=["is_mule"])     # feature không chứa label

    feats["train"], feats["val"], feats["test"], scaler = scale_features(
        feats["train"], feats["val"], feats["test"]
    )

    for name in ["train", "val", "test"]:              # gắn label lại SAU scale
        feats[name]["is_mule"] = labels[name]

    save_feature(feats["train"], feats["val"], feats["test"])
    for name in ["train", "val", "test"]:
        f = feats[name]
        n_mule = int(f["is_mule"].sum())
        print(f"  {name:5s}: {len(f):>6,} nodes | {n_mule:>5,} mule ({n_mule/len(f)*100:.2f}%)")

if __name__=="__main__":
    main()
                                     