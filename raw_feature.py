import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    df=pd.read_csv("dataset_high/HI-Small_Trans.csv",dtype={"From Bank":str, "Account":str, "To Bank": str, "Account.1":str})
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
        time_min=("Timestamp","min"),
        time_max=("Timestamp","max"),
        cross_bank_ratio=("is_cross_bank","mean"),
        cross_currency_ratio=("is_currency_change","mean"),
        round_amount_ratio=("is_round_amount","mean")
    )
    receiver=df_split.groupby("dest").agg(
        tong_nhan=("Amount Received","sum"),
        num_receive=("Amount Received","size"),
        mean_receive=("Amount Received","mean"),
        std_receive=("Amount Received","std"),
        num_bank_in=("From Bank","nunique"),
        currency_mix_in=("Receiving Currency","nunique"),
        in_degree=("src","nunique")
    )
    node_feature=sender.join(receiver, how="outer")
    node_feature["net_flow"]=node_feature["tong_nhan"].fillna(0) - node_feature["tong_gui"].fillna(0)
    node_feature["active_day"]=(node_feature["time_max"]-node_feature["time_min"]).dt.days
    node_feature["active_day"]=node_feature["active_day"].fillna(0).clip(lower=1)
    node_feature["tx_per_day"]=(node_feature["num_send"].fillna(0)+node_feature["num_receive"].fillna(0))/node_feature["active_day"]
    node_feature=node_feature.fillna(0)
    final_cols=["tong_gui","num_send","mean_send","std_send","num_bank_out","currency_mix_out","out_degree","cross_bank_ratio","round_amount_ratio",
                "cross_currency_ratio",
                "tong_nhan","num_receive","mean_receive","std_receive","num_bank_in","currency_mix_in","in_degree",
                "net_flow","active_day","tx_per_day"]
    return node_feature[final_cols]





