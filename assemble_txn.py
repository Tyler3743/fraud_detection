
import pandas as pd

DATA_DIR = "dataset_high"
TX_PATH = f"{DATA_DIR}/transaction_features.parquet"
LEAK_COLS_NODE = ["is_mule"]
LEAK_COLS_EDGE = ["is_edge_mule"]


def normalize_key(keys):
    parts = keys.str.split(" | ", regex=False)
    bank = parts.str[0].str.lstrip("0").replace("", "0")
    return bank + " | " + parts.str[1]


def load_node_features(split):
    nf = pd.read_csv(f"{DATA_DIR}/node_features_{split}.csv", index_col="node")
    nf.index = normalize_key(nf.index.to_series())
    return nf.drop(columns=LEAK_COLS_NODE, errors="ignore").astype("float32")


def load_edge_features(split):
    ef = pd.read_csv(f"{DATA_DIR}/edge_features_{split}.csv", index_col=["src", "dest"])
    ef = ef.drop(columns=LEAK_COLS_EDGE, errors="ignore").astype("float32")
    ef.columns = [f"e_{c}" for c in ef.columns]
    return ef


def assemble_split(tx, split):
    """Join node feat 2 dau + edge feat cho cac giao dich cua 1 split."""
    part = tx[tx["split"] == split].copy()
    nf = load_node_features(split)
    ef = load_edge_features(split)

    part = part.join(nf.add_prefix("src_"), on="src")
    part = part.join(nf.add_prefix("dest_"), on="dest")
    part = part.join(ef, on=["src", "dest"])

    y = part["Is Laundering"].astype("int8")
    X = part.drop(columns=["src", "dest", "Is Laundering", "split"])
    X = X.astype("float32").fillna(0)  # phong node/edge thieu (khong ky vong xay ra)
    X["Is Laundering"] = y
    return X


def main():
    tx = pd.read_parquet(TX_PATH)
    for split in ["train", "val", "test"]:
        mat = assemble_split(tx, split)
        out = f"{DATA_DIR}/txn_matrix_{split}.parquet"
        mat.to_parquet(out, index=False)
        n_pos = int(mat["Is Laundering"].sum())
        print(f"{split:5s}: {len(mat):>9,} rows | {mat.shape[1]-1} features | "
              f"{n_pos:,} pos ({n_pos/len(mat)*100:.3f}%) -> {out}")


if __name__ == "__main__":
    main()