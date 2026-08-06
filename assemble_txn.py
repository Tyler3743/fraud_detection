import pandas as pd, numpy as np, gc
import pyarrow.parquet as pq

data_dir  = "dataset_high"
tx_path   = f"{data_dir}/transaction_features.parquet"
asof_path = f"{data_dir}/node_edge_features.parquet"
label     = "Is Laundering"
key_cols  = ["src", "dest"]
leak      = ["is_mule", "is_edge_mule"]

def feat_cols(path, drop):
    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in drop]

def main():
    tx_cols   = feat_cols(tx_path,   set(key_cols + [label, "split"]))
    asof_cols = feat_cols(asof_path, {"split", "ori_idx"})

    # dup = sorted(set(tx_cols) & set(asof_cols))
    # if dup: raise AssertionError(f"trùng tên cột giữa hai file: {dup}")
    # bad = [c for c in leak if c in tx_cols or c in asof_cols]
    # if bad: raise AssertionError(f"cột suy từ nhãn lọt vào feature: {bad}")
    meta_tx   = pd.read_parquet(tx_path,   columns=[label, "split"])
    # meta_asof = pd.read_parquet(asof_path, columns=["split"])
    # if len(meta_tx) != len(meta_asof):
    #     raise AssertionError(f"số dòng lệch: {len(meta_tx):,} vs {len(meta_asof):,}")
    # n_bad = int((meta_tx["split"].values != meta_asof["split"].values).sum())
    # if n_bad: raise AssertionError(f"{n_bad:,} dòng lệch 'split'")
    # del meta_asof; gc.collect()

    y     = meta_tx[label].to_numpy(dtype="int8")
    split = meta_tx["split"].to_numpy()
    n     = len(y)
    del meta_tx; gc.collect()
    # print(f"[align] {n:,} dòng khớp thứ tự")

    names = tx_cols + asof_cols
    X = np.empty((n, len(names)), dtype="float32")
    j = 0
    for path, cols in ((tx_path, tx_cols), (asof_path, asof_cols)):
        pf = pq.ParquetFile(path)
        for c in cols:
            v = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)
            v = v.astype("float32", copy=False)
            # if not np.isfinite(v).all():
            #     raise AssertionError(f"NaN/inf ở cột: {c}")
            X[:, j] = v; j += 1
            del v
        del pf; gc.collect()
    # print(f"[build] {len(tx_cols)} tx + {len(asof_cols)} as-of = {j} feature, không NaN/inf")

    for s in ["train", "val", "test"]:
        m   = split == s
        matrix = pd.DataFrame(X[m], columns=names)
        matrix[label] = y[m]
        out = f"{data_dir}/txn_matrix_{s}.parquet"
        matrix.to_parquet(out, index=False)
        n_pos = int(matrix[label].sum())
        print(f"{s:5s}: {len(matrix):>9,} dòng | {len(names)} feature | "
              f"{n_pos:,} pos ({n_pos/len(matrix)*100:.3f}%) -> {out}")
        del matrix; gc.collect()

if __name__ == "__main__":
    main()