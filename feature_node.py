import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

src="dataset_high/HI-Small_Trans_split_index.csv"
out="dataset_high/node_edge_features"

def load_data():
    dt={"From Bank": str, "Account": str, "To Bank": str, "Account.1": str, "split": str}
    df=pd.read_csv(src,dtype=dt,engine="pyarrow")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    df["src"]=df["From Bank"]+ " | " +df["Account"]
    df["dest"]=df["To Bank"]+ " | " +df["Account.1"]
    df["ori_idx"]=np.arange(len(df))
    df=df.sort_values(["Timestamp","ori_idx"], kind="mergesort").reset_index(drop=True)
    df["time"]=df["Timestamp"].astype("int64")
    df["is_cross_bank"]=df["From Bank"]!=df["To Bank"]
    df["is_cross_curcy"]=df["Payment Currency"]!=df["Receiving Currency"]
    # df["is_round"]=df["Amount Paid"]%1000==0
    return df

def asof_block(df, group_cols, amt_col, partner_col, bank_col, curcy_col,pf_encoding,prefix,log_col_out):
    #mean/sum/count
    out=pd.DataFrame(index=df.index)
    keys=[df[c] for c in group_cols]
    g=df.groupby(group_cols,sort=False)
    x=df[amt_col]
    cnt=g.cumcount()
    out[f"{prefix}cnt"]=cnt
    sum_prior=g[amt_col].cumsum()-x
    out[f"{prefix}sum"]=sum_prior
    mean_prior=(sum_prior/cnt).fillna(0)
    out[f"{prefix}mean"]=mean_prior
    #phương sai
    x2=x**2
    sumsq_prior=x2.groupby(keys, sort=False).cumsum()-x2
    meansq_prior=(sumsq_prior/cnt).fillna(0)
    var_prior=meansq_prior-mean_prior**2
    std_prior = np.sqrt(np.maximum(var_prior, 0.0))
    out[f"{prefix}std"]=std_prior
    #min/max
    out[f"{prefix}max"]=g[amt_col].cummax().groupby(keys,sort=False).shift(1).fillna(0)
    out[f"{prefix}min"]=g[amt_col].cummin().groupby(keys,sort=False).shift(1).fillna(0)
    
    def nunique_prior(sub_col):
        flag=(~df.duplicated(subset=group_cols+[sub_col])).astype("float32")
        count=flag.groupby(keys,sort=False).cumsum()
        return count-flag
    if partner_col is not None:
        out[f"{prefix}cnt_partner"]=nunique_prior(partner_col)
    if bank_col is not None:
        out[f"{prefix}cnt_bank"]=nunique_prior(bank_col)
    if curcy_col is not None:
        out[f"{prefix}cnt_curcy"]=nunique_prior(curcy_col)
    for fname, fcol in [("cross_bank","is_cross_bank"),("cross_curcy","is_cross_curcy")]:#xóa is_round
        prior_f=df[fcol].groupby(keys,sort=False).cumsum()-df[fcol]
        out[f"{prefix}ratio_{fname}"]=(prior_f/cnt).fillna(0).astype("float32")
    for col in pf_encoding.columns:
        prior_d=pf_encoding[col].groupby(keys,sort=False).cumsum()-pf_encoding[col]
        out[f"{prefix}ratio_{col}"]=(prior_d/cnt).fillna(0).astype("float32")
    first_seen=df["time"].groupby(keys,sort=False).cummin()
    active_time=((df["time"]-first_seen)/1e9).fillna(0)
    out[f"{prefix}active_time"]=active_time
    out[f"{prefix}tx_per_day"]=(cnt/(active_time/86400).clip(lower=1.0)).fillna(0)
    out[f"{prefix}first_seen"]=(cnt==0).astype("int8")
    log_col_out+=[f"{prefix}cnt",f"{prefix}sum",f"{prefix}mean",f"{prefix}std",
                f"{prefix}max",f"{prefix}min",f"{prefix}tx_per_day",f"{prefix}active_time"]
    for c in (partner_col,curcy_col,bank_col):
        if c is not None:
            suffix={partner_col:"partner",bank_col:"bank",curcy_col:"curcy"}[c]
            log_col_out.append(f"{prefix}cnt_{suffix}")
    return out

def build_entity_features(df):
    pf_encoding=pd.get_dummies(df["Payment Format"],dtype="int8")
    log_cols=[]
    src_block=asof_block(df,["src"],"Amount Paid", partner_col="dest", bank_col="To Bank",curcy_col="Payment Currency",
                         pf_encoding=pf_encoding,prefix="src_",log_col_out=log_cols)
    dest_block=asof_block(df,["dest"],"Amount Received", partner_col="src", bank_col="From Bank",curcy_col="Receiving Currency",
                         pf_encoding=pf_encoding,prefix="dest_",log_col_out=log_cols)
    pair_block=asof_block(df,["src","dest"],"Amount Paid", partner_col=None, bank_col=None,curcy_col=None,
                         pf_encoding=pf_encoding,prefix="pair_",log_col_out=log_cols)
    asof=pd.concat([src_block,dest_block,pair_block],axis=1)
    asof["split"]=df["split"].values
    asof["ori_idx"]=df["ori_idx"].values
    return asof, log_cols
def verify_asof(df, asof):
    verify=True
    first_row=~df.duplicated(subset=["src"],keep="first")
    check_first=(asof.loc[first_row,"src_cnt"]==0).all() and (asof.loc[first_row,"src_first_seen"]== 1).all()\
                 and (asof.loc[first_row,"src_sum"]==0).all()
    print(f"kiểm tra rò rỉ dữ liệu: {check_first} {'PASS' if check_first else 'lỗi check first'}")
    verify&=check_first
    last_row=~df.duplicated(subset=["src"],keep="last")
    whole_window_count=df.groupby("src")["src"].transform("size")
    check_last=(asof.loc[last_row,"src_cnt"]==whole_window_count.loc[last_row]-1).all()
    print(f"kiểm tra toàn vẹn: {'PASS' if check_last else 'lỗi check last'}")
    verify&=check_last
    test_rows=df.index[df["split"]=="test"]
    idx=test_rows[len(test_rows)//2]
    row=df.loc[idx]
    prior_mask=(df["src"]==row["src"]) & ((df["time"]<row["time"]) | \
                    ((df["time"]==row["time"]) & (df["ori_idx"]<row["ori_idx"])))
    manual_cnt=int(prior_mask.sum())
    manual_sum=df.loc[prior_mask,"Amount Paid"].sum()
    got_cnt=asof.loc[idx,"src_cnt"]
    got_sum=asof.loc[idx,"src_sum"]
    spot_check=(manual_cnt==got_cnt) and np.isclose(manual_sum,got_sum)
    print(f"kiểm tra ngẫu nhiên: index={idx}: count ={manual_cnt} vs count.1={got_cnt},"\
          f"sum={manual_sum:.2f} vs sum.1={got_sum:.2f}")
    verify&=spot_check
    if not verify:
        raise AssertionError("check fail sau khi kiểm tra logic")
    print("PASS ALL")
def scale_feature(asof,log_cols):
    out=asof.copy()
    out[log_cols]=np.log1p(out[log_cols].clip(lower=0))
    is_train=out["split"]=="train"
    scaler=StandardScaler().fit(out.loc[is_train,log_cols].values)
    out[log_cols]=scaler.transform(out[log_cols].values)
    return out, scaler

def main():
    df=load_data()
    asof, log_cols=build_entity_features(df)
    verify_asof(df,asof)
    asof,scaler=scale_feature(asof,log_cols)
    asof = asof.sort_values("ori_idx").drop(columns="ori_idx")
    os.makedirs("dataset_high", exist_ok=True)
    path = out+".parquet"
    asof.to_parquet(path,index=False)
    print(f"Success Đã lưu: {path} shape={asof.shape}")
if __name__=="__main__":
    main()