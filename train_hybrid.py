# train_hybrid.py — nhóm 3: embedding GraphSAGE-LSTM + gradient boosting
import json, os, time
import numpy as np, pandas as pd, joblib
import pyarrow.parquet as pq
from paths import TX_FEAT, TXN_NODES, TXN_MATRIX, OUT_DIR
from metrics import find_best_threshold, evaluate, log_result
from train_classical import PARAMS, build_model, fit   # tái dùng siêu tham số + early-stopping nhóm 1

EMB_DIR  = "scores/emb-ssl-v3"
LABEL    = "Is Laundering"
SPLITS   = ["train", "val", "test"]
SEEDS    = [0, 1, 2, 3]
BOOSTERS = ["lgb", "xgb"]
EMB_DIM  = 32
N_SCALAR = 3                            
CHUNK    = 200_000                     
RESULTS  = "results-hybrid-v3.csv"
DEVICE   = "local-i5-13420H"

P1_EXTRA = N_SCALAR                                 
ARMS_P1  = [("V0",  None,               0,        False),   
            ("V1e", "ssl_lstm",         N_SCALAR, False)]  

P2_EXTRA = 2 * EMB_DIM + N_SCALAR                    
ARMS_P2  = [("V2e", "ssl_lstm",         2 * EMB_DIM,            True),   
            ("V3e", "ssl_lstm",         2 * EMB_DIM + N_SCALAR, True),  
            ("V2n", "ssl_noedge_lstm",  2 * EMB_DIM,            True)]   


def alloc(n_extra):
    split  = pd.read_parquet(TX_FEAT, columns=["split"])["split"].to_numpy()
    nodes  = np.load(TXN_NODES)                                  # [5_078_345, 2]
    mask_b = np.load(f"{OUT_DIR}/train_b_mask.npy")[split == "train"]

    buf, ys, pairs, n_base = {}, {}, {}, None
    for s in SPLITS:
        pf   = pq.ParquetFile(TXN_MATRIX.format(s))
        cols = [c for c in pf.schema_arrow.names if c != LABEL]
        keep = mask_b if s == "train" else slice(None)

        p_all = nodes[split == s]
        assert pf.metadata.num_rows == len(p_all), f"{s}: txn_matrix lệch txn_nodes"
        p = p_all[keep]
        if n_base is None:
            n_base = len(cols)
        assert len(cols) == n_base, "số cột gốc khác nhau giữa các split"

        X = np.empty((len(p), n_base + n_extra), dtype="float32")
        for j, c in enumerate(cols):
            v = pf.read(columns=[c]).column(0).to_numpy(zero_copy_only=False)
            X[:, j] = v[keep]
            del v
        y = pf.read(columns=[LABEL]).column(0).to_numpy(zero_copy_only=False)[keep]

        buf[s], ys[s], pairs[s] = X, y.astype("int8"), p
        print(f"{s:5s}: {len(p):>9,} dòng | buffer {X.nbytes/1e9:.2f} GB | pos {int(y.sum()):,}")
    return buf, ys, pairs, n_base


def require_emb():
    need = [f"{EMB_DIR}/emb_{e}_seed{s}_{sp}.npy"
            for e in {a[1] for a in ARMS_P1 + ARMS_P2 if a[1]} for s in SEEDS for sp in SPLITS]
    miss = [p for p in need if not os.path.exists(p)]
    assert not miss, (f"thiếu {len(miss)}/{len(need)} file embedding")
    print(f"embedding: đủ {len(need)} file trong {EMB_DIR}")


def fill_emb(buf, pairs, encoder, seed, n_base, write_raw):
    es = n_base + (2 * EMB_DIM if write_raw else 0)      # vị trí khối 3 vô hướng
    for s in SPLITS:
        h = np.load(f"{EMB_DIR}/emb_{encoder}_seed{seed}_{s}.npy")
        assert h.shape[1] == EMB_DIM and pairs[s].max() < len(h), f"{s}: embedding lệch"
        X, p = buf[s], pairs[s]
        for i in range(0, len(p), CHUNK):
            sl = slice(i, i + CHUNK)
            u, v = h[p[sl, 0]], h[p[sl, 1]]
            if write_raw:
                X[sl, n_base:n_base + EMB_DIM] = u
                X[sl, n_base + EMB_DIM:es]     = v
            dot = np.einsum("ij,ij->i", u, v)               # không dựng mảng tích trung gian
            nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
            X[sl, es]     = dot / (nu * nv + 1e-8)          # cosine
            X[sl, es + 1] = np.linalg.norm(u - v, axis=1)   # L2
            X[sl, es + 2] = dot
            del u, v, dot, nu, nv
        del h


def run_arm(arm, encoder, ncol, buf, ys, seed, rows):
    pos_w = (ys["train"] == 0).sum() / ys["train"].sum()  
    for name in BOOSTERS:                                 
        tag   = f"{arm}_{name}"
        model = build_model(name, seed, pos_w)
        t0 = time.time(); fit(model, name, buf["train"][:, :ncol], ys["train"])
        tt = round(time.time() - t0, 1)

        sc = {s: model.predict_proba(buf[s][:, :ncol])[:, 1] for s in ("val", "test")}
        for s in ("val", "test"):
            np.save(f"scores/{tag}_seed{seed}_{s}.npy", sc[s])

        thr  = find_best_threshold(ys["val"], sc["val"])
        meta = json.dumps({**PARAMS[name], "device": DEVICE, "encoder": encoder,
                           "emb_seed": seed if encoder else None,
                           "n_feature": int(ncol),
                           "train_rows": int(len(ys["train"]))})
        for s in ("val", "test"):
            m = evaluate(ys[s], sc[s], thr)
            log_result(tag, s, m, path=RESULTS, seed=seed, train_time_s=tt,
                       stage="final", params=meta)
            rows.append({"arm": arm, "booster": name, "seed": seed, "split": s,
                         "f1": m["f1_minority"], "pr_auc": m["pr_auc"]})
        (model.save_model(f"models/{tag}_seed{seed}.json") if name == "xgb"
         else joblib.dump(model, f"models/{tag}_seed{seed}.joblib"))
        print(f"[{tag:8s} seed{seed}] {ncol:>3d} cột | {tt:>6.1f}s | f1_test={rows[-1]['f1']:.4f}")


def main():
    os.makedirs("scores", exist_ok=True); os.makedirs("models", exist_ok=True)
    require_emb()
    if os.path.exists(RESULTS):
        os.remove(RESULTS)          
    rows = []

    for extra, arms in [(P1_EXTRA, ARMS_P1), (P2_EXTRA, ARMS_P2)]:
        buf, ys, pairs, n_base = alloc(extra)
        for seed in SEEDS:
            loaded = None                       # (encoder, write_raw) đang nằm trong buffer
            for arm, encoder, n_use, write_raw in arms:
                if encoder is not None and (encoder, write_raw) != loaded:
                    fill_emb(buf, pairs, encoder, seed, n_base, write_raw)
                    loaded = (encoder, write_raw)
                run_arm(arm, encoder, n_base + n_use, buf, ys, seed, rows)
        del buf, ys, pairs

    r = pd.DataFrame(rows); r = r[r.split == "test"]
    print("\n=== test, mean ± std trên 4 seed ===")
    print(r.groupby(["booster", "arm"])[["f1", "pr_auc"]].agg(["mean", "std"]).round(4))

    w = r.pivot_table(index=["booster", "seed"], columns="arm", values="f1")
    print("\n=== hiệu ghép cặp theo seed (f1_minority) ===")
    print(pd.DataFrame({
        "V2e-V0":  w["V2e"] - w["V0"],    # embedding thô đóng góp bao nhiêu
        "V1e-V0":  w["V1e"] - w["V0"],    # chỉ 3 vô hướng đóng góp bao nhiêu
        "V2e-V1e": w["V2e"] - w["V1e"],   # 64 chiều thô có hơn 3 vô hướng không
        "V3e-V2e": w["V3e"] - w["V2e"],   # boosting đã khai thác hết embedding thô chưa
        "V2e-V2n": w["V2e"] - w["V2n"],   # edge_attr còn giá trị dưới head boosting không
    }).round(4))


if __name__ == "__main__":
    main()