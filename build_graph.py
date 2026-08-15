import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
#tạo ra txn_node và graphs.zip
from feature_node import load_data
from paths import EDGE_ATTR, GRAPHS_PT, TXN_NODES

GRAPH_SPLITS = {                       # cấu trúc lũy tiến chuẩn Altman
    "train": ["train"],
    "val":   ["train", "val"],
    "test":  ["train", "val", "test"],
}


def build_vocab(df):
    """Vocab node DÙNG CHUNG cho cả 3 graph -> embedding khớp index giữa các split."""
    a = df[["src", "From Bank"]].rename(columns={"src": "node", "From Bank": "bank"})
    b = df[["dest", "To Bank"]].rename(columns={"dest": "node", "To Bank": "bank"})
    nodes = (pd.concat([a.drop_duplicates("node"), b.drop_duplicates("node")])
               .drop_duplicates("node").reset_index(drop=True))
    node_id = pd.Series(np.arange(len(nodes), dtype=np.int64), index=nodes["node"])
    bank_code, banks = pd.factorize(nodes["bank"])
    x = torch.from_numpy(bank_code.astype(np.int64)).view(-1, 1)   # nn.Embedding trong model
    return node_id, x, len(banks)


def build_graph(name, node_id, x):
    ea = pd.read_parquet(EDGE_ATTR.format(name))
    si = node_id.reindex(ea["src"]).to_numpy()
    di = node_id.reindex(ea["dest"]).to_numpy()
    assert not (np.isnan(si).any() or np.isnan(di).any()), "cạnh có node ngoài vocab"

    feat = ea.drop(columns=["src", "dest"]).to_numpy(dtype=np.float32)
    fwd = torch.from_numpy(np.stack([si, di]).astype(np.int64))
    rev = fwd.flip(0)                                   # cạnh ngược, cùng edge_attr
    attr = torch.from_numpy(feat)
    flag = torch.zeros(attr.size(0), 1)

    data = Data(
        x=x,
        edge_index=torch.cat([fwd, rev], dim=1),
        edge_attr=torch.cat([torch.cat([attr, flag], 1),
                             torch.cat([attr, flag + 1], 1)], dim=0),   # +cờ is_reverse
        num_nodes=x.size(0),
    )
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
    print(f"  {name:5s}: {data.num_nodes:,} node | {fwd.size(1):,} cạnh có hướng "
          f"-> {data.edge_index.size(1):,} sau cạnh ngược | "
          f"edge_attr {tuple(data.edge_attr.shape)} | node cô lập {int((deg == 0).sum()):,}")
    return data


def main():
    df = load_data()
    node_id, x, n_bank = build_vocab(df)
    print(f"vocab: {len(node_id):,} node | {n_bank:,} bank")

    graphs = {n: build_graph(n, node_id, x) for n in GRAPH_SPLITS}
    os.makedirs(os.path.dirname(GRAPHS_PT), exist_ok=True)
    torch.save({"graphs": graphs, "num_banks": n_bank}, GRAPHS_PT)

    # ánh xạ giao dịch -> (node gửi, node nhận), theo ĐÚNG thứ tự file gốc
    df = df.sort_values("ori_idx")
    txn = np.stack([node_id.reindex(df["src"]).to_numpy(),
                    node_id.reindex(df["dest"]).to_numpy()]).astype(np.int64).T
    np.save(TXN_NODES, txn)
    print(f"đã lưu {GRAPHS_PT} và {TXN_NODES} shape={txn.shape}")


if __name__ == "__main__":
    main()