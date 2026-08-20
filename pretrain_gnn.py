import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from node_seq import NODE_SEQ
from paths import GRAPHS_PT, SCORES_DIR, TXN_MATRIX, WORK_DIR
from train_gnn import DEV, HID, Model          

SEEDS    = [0, 1, 2, 3]
VARIANTS = ["ssl_lstm", "ssl_noedge_lstm"]    
EMB_DIR  = os.path.join(SCORES_DIR, "emb-ssl")
BOARD    = os.path.join(WORK_DIR, "pretrain_board-v3.csv")

MP_FRAC, SUP_FRAC = 0.85, 0.10   
STEPS       = 600
EDGE_BATCH  = 65_536
LR          = 3e-3
EVAL_EVERY  = 20
PATIENCE    = 20                  
NEG_POW     = 0.75               


def load_graphs():
    graph = torch.load(GRAPHS_PT, weights_only=False)# load 3 đồ thị train, val, test
    graphs, num_banks = graph["graphs"], graph["num_banks"]
    for s, g in graphs.items():
        g.node_seq = torch.from_numpy(np.load(NODE_SEQ.format(s)))
        graphs[s] = g.to(DEV) #
    tx_dim = len(pq.ParquetFile(TXN_MATRIX.format("train")).schema_arrow.names) - 1
    return graphs, num_banks, tx_dim #đồ thị, lập bank ID, số chiều của mỗi đặc trưng


def split_edges(g, seed):
    E = g.edge_index.size(1) // 2
    ei = g.edge_index[:, :E].cpu().numpy().astype(np.int64)
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    uniq, inv = np.unique(lo * g.num_nodes + hi, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(uniq)) #permutation: hoán vị
    n_mp, n_sup = int(len(uniq) * MP_FRAC), int(len(uniq) * SUP_FRAC) #number of message passing= số lượng cạnh truyền tin
    #number supervision= số lượng cạnh giám sát
    bucket = np.empty(len(uniq), dtype=np.int8)
    bucket[perm[:n_mp]] = 0
    bucket[perm[n_mp:n_mp + n_sup]] = 1
    bucket[perm[n_mp + n_sup:]] = 2
    b = bucket[inv]
    mp, sup, ev = (np.flatnonzero(b == k) for k in (0, 1, 2))
    assert not (set(uniq[bucket == 0]) & set(uniq[bucket != 0])), "rổ cạnh chồng nhau"

    dev = g.edge_index.device
    keep = torch.from_numpy(np.concatenate([mp, mp + E])).long().to(dev)
    g_mp = Data(x=g.x, edge_index=g.edge_index[:, keep], edge_attr=g.edge_attr[keep],
                num_nodes=g.num_nodes)
    g_mp.node_seq = g.node_seq

    print(f"  cạnh gốc {E:,} ({len(uniq):,} cặp vô hướng) -> truyền tin {len(mp):,} | "
          f"giám sát {len(sup):,} | dừng sớm {len(ev):,}")
    return (g_mp,
            g.edge_index[:, torch.from_numpy(sup).long().to(dev)],
            g.edge_index[:, torch.from_numpy(ev).long().to(dev)])


def neg_table(g_mp, n_node):
    deg = torch.bincount(g_mp.edge_index[0], minlength=n_node).double().cpu().numpy()
    p = deg ** NEG_POW
    cdf = np.cumsum(p / p.sum())
    cdf[-1] = 1.0
    return cdf


def sample_neg(cdf, n, rng):
    return np.minimum(np.searchsorted(cdf, rng.random(n)), len(cdf) - 1)


@torch.no_grad()
def eval_ap(model, g_mp, pos, neg_dst):
    model.eval()
    h = model.encode(g_mp)
    sp = (h[pos[0]] * h[pos[1]]).sum(-1).float().cpu().numpy()
    sn = (h[pos[0]] * h[neg_dst]).sum(-1).float().cpu().numpy()
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    return float(average_precision_score(y, np.concatenate([sp, sn])))


def pretrain(tag, seed, graphs, num_banks, tx_dim):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = graphs["train"]
    g_mp, sup, ev = split_edges(g, seed)
    cdf = neg_table(g_mp, g.num_nodes)

    model = Model(num_banks, g.edge_attr.size(1), tx_dim, g.node_seq.size(-1),
                  hid=HID, use_edge=(tag == "ssl_lstm")).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    amp = DEV.type == "cuda"
    gscaler = torch.amp.GradScaler("cuda", enabled=amp)

    ev_neg = torch.from_numpy(
        sample_neg(cdf, ev.size(1), np.random.default_rng(12345))).long().to(DEV)

    best, best_state, bad, step, t0 = -1.0, None, 0, 0, time.time()
    for step in range(STEPS):
        model.train()
        b = torch.from_numpy(rng.integers(0, sup.size(1), EDGE_BATCH)).long().to(DEV)
        u, v = sup[0][b], sup[1][b]
        nd = torch.from_numpy(sample_neg(cdf, EDGE_BATCH, rng)).long().to(DEV)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            h = model.encode(g_mp)
            pos = (h[u] * h[v]).sum(-1)
            neg = (h[u] * h[nd]).sum(-1)
            loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                    + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
        gscaler.scale(loss).backward()
        gscaler.step(opt)
        gscaler.update()

        if (step + 1) % EVAL_EVERY == 0:
            ap = eval_ap(model, g_mp, ev, ev_neg)
            if ap > best:
                best, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  step{step + 1:4d} loss={loss.item():.4f} link_ap={ap:.4f} "
                  f"(best {best:.4f}, bad {bad}/{PATIENCE})")
            if bad >= PATIENCE:
                break

    if best_state is not None:                      
        model.load_state_dict(best_state)
    model.eval()
    train_time_s = round(time.time() - t0, 1)

    os.makedirs(EMB_DIR, exist_ok=True)
    with torch.no_grad():
        for s in ["train", "val", "test"]:
            h = model.encode(graphs[s]).float().cpu().numpy()
            assert np.isfinite(h).all(), f"{tag} seed{seed} {s}: embedding có NaN/inf"
            np.save(os.path.join(EMB_DIR, f"emb_{tag}_seed{seed}_{s}.npy"), h)
    print(f"[{tag} seed{seed}] {train_time_s}s | link_ap={best:.4f} | đã xuất 3 embedding")
    return {"model": tag, "seed": seed, "link_ap": round(best, 4),
            "steps_run": step + 1, "train_time_s": train_time_s}


def main():
    print(f"device={DEV} | variants={VARIANTS} | seeds={SEEDS}")
    graphs, num_banks, tx_dim = load_graphs()
    print(f"tx_dim={tx_dim} | seq_dim={graphs['train'].node_seq.size(-1)} | "
          f"banks={num_banks:,}")

    rows = [pretrain(tag, seed, graphs, num_banks, tx_dim)
            for tag in VARIANTS for seed in SEEDS]

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(BOARD, index=False)
    print(f"\n=== link prediction AP ({len(SEEDS)} seed) ===")
    print(df.groupby("model")["link_ap"].agg(["mean", "std"]).round(4))
    print(f"\nembedding -> {EMB_DIR}")
    print(json.dumps({"emb_dir": EMB_DIR, "variants": VARIANTS, "seeds": SEEDS,
                      "objective": "link_prediction", "uses_label": False}, indent=2))
    if DEV.type == "cuda":
        print(f"VRAM đỉnh: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()