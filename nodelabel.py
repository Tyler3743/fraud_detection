import pandas as pd

df=pd.read_csv("dataset_high/HI-Small_Trans.csv")
fraud=df[df["Is Laundering"]==1]

all_accounts= pd.concat([df['Account'],df['Account.1']]).unique()#gom tất cả tài khoản
#lọc ra dòng có Is Laundering=1
IsFraud = df[df["Is Laundering"]==1]
# lọc các tài khoản gian lận, dính tới Laundering=1
account_fraud = set(pd.concat([IsFraud["Account"],IsFraud["Account.1"]]).unique())
# tạo bảng và gán nhãn
node_labels=pd.DataFrame({
    "Account": list(all_accounts)
})
node_labels["IsLaundering"]=node_labels["Account"].apply(
    lambda x:1 if x in account_fraud else 0
)
# xuất file
node_labels.to_csv("AccountFraudLabel.csv", index=False)
