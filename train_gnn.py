import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import MessagePassing
from node_seq import NODE_SEQ
from metrics import find_best_threshold, evaluate, log_result
from paths import (TX_FEAT, TXN_NODES, GRAPHS_PT, TXN_MATRIX, SCORES_DIR, OUT_DIR,
                   WORK_DIR, RESULTS_CSV)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3]
HID, NEG_RATIO, EPOCHS, BATCH, LR, PATIENCE = 16, 50, 200, 8192, 3e-3, 40
SEQ_CHUNK = 64_000                     # chia số node cho LSTM

VARIANTS = ["sage_lstm", "sage_noedge_lstm"]     # cùng kiến trúc, chỉ khác có/không edge_attr
SELECT_SPLIT, SELECT_METRIC = "val", "pr_auc"   # chọn model: PR-AUC trên VAL
REPORT = ["pr_auc", "f1_minority", "f1@0.5", "recall"]

TRAIN_B_MASK = os.path.join(OUT_DIR, "train_b_mask.npy")
WINNER_JSON  = os.path.join(WORK_DIR, "winner-v3.json")
BOARD_CSV    = os.path.join(WORK_DIR, "variant_board-v3.csv")


class EdgeSAGE(MessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim, use_edge=True):
        super().__init__(aggr=["mean", "max"])   # max là điều kiện để port number có tác dụng
        self.use_edge = use_edge
        self.msg = nn.Linear(in_dim + edge_dim if use_edge else in_dim, out_dim)
        self.root = nn.Linear(in_dim, out_dim)
        self.mix = nn.Linear(2 * out_dim, out_dim)                     # gộp [mean ‖ max]

    def forward(self, h, edge_index, edge_attr):
        return self.root(h) + self.mix(self.propagate(edge_index, x=h, edge_attr=edge_attr))

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=-1) if self.use_edge else x_j
        return F.relu(self.msg(z))


class Model(nn.Module):
    def __init__(self, num_banks, edge_dim, tx_dim, seq_dim, hid=HID, use_edge=True):
        super().__init__()
        self.bank = nn.Embedding(num_banks, hid)
        self.seq = nn.LSTM(seq_dim, hid, num_layers=2, batch_first=True)   # Temporal Encoder
        self.c1 = EdgeSAGE(2 * hid, edge_dim, hid, use_edge)               # bank_emb ‖ lstm
        self.c2 = EdgeSAGE(hid, edge_dim, hid, use_edge)
        self.head = nn.Sequential(
            nn.Linear(2 * hid + tx_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def _lstm_last(self, s):
        return self.seq(s)[1][0][-1]                                       # h_n lớp cuối: [B, hid]

    def _temporal(self, seq):
        out = []
        for i in range(0, seq.size(0), SEQ_CHUNK):
            s = seq[i:i + SEQ_CHUNK]
            out.append(cp.checkpoint(self._lstm_last, s, use_reentrant=False)
                       if self.training else self._lstm_last(s))
        return torch.cat(out, 0)

    def encode(self, g):
        h = torch.cat([self.bank(g.x.squeeze(-1)), self._temporal(g.node_seq)], dim=-1)
        h = F.relu(self.c1(h, g.edge_index, g.edge_attr))
        return self.c2(h, g.edge_index, g.edge_attr)

    def forward(self, h, pairs, tx):
        return self.head(torch.cat([h[pairs[:, 0]], h[pairs[:, 1]], tx], -1)).squeeze(-1)


def load_all():
    split = pd.read_parquet(TX_FEAT, columns=["split"])["split"].to_numpy()
    nodes = np.load(TXN_NODES)
    blob = torch.load(GRAPHS_PT, weights_only=False)
    graphs, num_banks = blob["graphs"], blob["num_banks"]

    d = {}
    for s in ["train", "val", "test"]:
        m = pd.read_parquet(TXN_MATRIX.format(s))
        y = m.pop("Is Laundering").to_numpy().astype("float32")
        d[s] = [m.to_numpy("float32"), y, torch.from_numpy(nodes[split == s])]

    sc = StandardScaler().fit(d["train"][0])                 # fit CHỈ trên train
    for s in d:
        d[s][0] = sc.transform(d[s][0]).astype("float32")

    for s, g in graphs.items():
        g.node_seq = torch.from_numpy(np.load(NODE_SEQ.format(s)))
        graphs[s] = g.to(DEV)

    # CHỈ train trên train_b: thuộc tính cạnh của train-graph tính từ train_a (lagged)
    pool = np.flatnonzero(np.load(TRAIN_B_MASK)[split == "train"])
    ytr = d["train"][1]
    print(f"train_b: {len(pool):,}/{len(ytr):,} dòng | positive {int(ytr[pool].sum()):,} "
          f"(toàn train {int(ytr.sum()):,})")
    return d, graphs, num_banks, pool


def sample_epoch(y, rng, pool):
    pos = pool[y[pool] == 1]
    neg = pool[y[pool] == 0]
    neg = rng.choice(neg, size=min(len(neg), NEG_RATIO * len(pos)), replace=False)
    idx = np.concatenate([pos, neg]); rng.shuffle(idx)
    return idx


@torch.no_grad()
def score(model, g, X, pairs, chunk=200_000):
    model.eval()
    h = model.encode(g)
    out = [torch.sigmoid(model(h, pairs[i:i + chunk].to(DEV),
                               torch.from_numpy(X[i:i + chunk]).to(DEV))).float().cpu().numpy()
           for i in range(0, len(X), chunk)]
    return np.concatenate(out), h

def run(tag, seed, d, graphs, num_banks, pool):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr, ytr, ptr = d["train"]
    model = Model(num_banks, graphs["train"].edge_attr.size(1), Xtr.shape[1],
                  graphs["train"].node_seq.size(-1),
                  use_edge=(tag == "sage_lstm")).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    amp = DEV.type == "cuda"
    gscaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(EPOCHS):
        idx = sample_epoch(ytr, rng, pool)
        model.train()
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                h = model.encode(graphs["train"])
                logit = model(h, ptr[b].to(DEV), torch.from_numpy(Xtr[b]).to(DEV))
                loss = F.binary_cross_entropy_with_logits(
                    logit, torch.from_numpy(ytr[b]).to(DEV))
            gscaler.scale(loss).backward(); gscaler.step(opt); gscaler.update()

        s_val, _ = score(model, graphs["val"], d["val"][0], d["val"][2])
        pr = evaluate(d["val"][1], s_val, 0.5)["pr_auc"]
        if pr > best:
            best, bad = pr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
        if ep % 10 == 0:
            print(f"  ep{ep:3d} loss={loss.item():.4f} val_pr_auc={pr:.4f} (best {best:.4f})")

    model.load_state_dict(best_state)
    train_time_s = round(time.time() - t0, 1)

    s_val, _ = score(model, graphs["val"], d["val"][0], d["val"][2])
    s_test, _ = score(model, graphs["test"], d["test"][0], d["test"][2])
    thr = find_best_threshold(d["val"][1], s_val)           # ngưỡng dò trên VAL

    rows = []
    for s, sc_ in [("val", s_val), ("test", s_test)]:
        m = evaluate(d[s][1], sc_, thr)
        log_result(tag, s, m, path=RESULTS_CSV, seed=seed, stage="final",
                   train_time_s=train_time_s,
                   params=json.dumps({"hid": HID, "neg_ratio": NEG_RATIO, "batch": BATCH,
                                      "patience": PATIENCE, "epochs_run": ep + 1,
                                      "train_rows": len(pool)}))
        np.save(os.path.join(SCORES_DIR, f"{tag}_seed{seed}_{s}-v3.npy"), sc_)
        rows.append({"model": tag, "seed": seed, "split": s, **m})
    print(f"[{tag} seed{seed}] {train_time_s}s | val_pr_auc={best:.4f}")
    return rows, best_state

def main():
    os.makedirs(SCORES_DIR, exist_ok=True)
    print(f"device={DEV} | variants={VARIANTS} | seeds={SEEDS}")
    d, graphs, num_banks, pool = load_all()

    rows = []
    for tag in VARIANTS:
        for seed in SEEDS:
            r, _ = run(tag, seed, d, graphs, num_banks, pool)
            rows += r

    df = pd.DataFrame(rows)
    board = df.groupby(["split", "model"])[REPORT].agg(["mean", "std"]).round(4)
    board.to_csv(BOARD_CSV)
    print(f"\n=== bảng so biến thể ({len(SEEDS)} seed) ===\n{board}\n")

    sel = df[df.split == SELECT_SPLIT].groupby("model")[SELECT_METRIC].mean()
    winner = sel.idxmax()
    print(f"CHỌN: {winner}  (tiêu chí {SELECT_METRIC} trên {SELECT_SPLIT}: "
          + ", ".join(f"{k}={v:.4f}" for k, v in sel.items()) + ")")

    json.dump({"winner": winner, "variants": VARIANTS, "seeds": SEEDS,
               "select_on": f"{SELECT_SPLIT}/{SELECT_METRIC}",
               "select_scores": {k: round(float(v), 4) for k, v in sel.items()}},
              open(WINNER_JSON, "w"), indent=2)
    if DEV.type == "cuda":
        print(f"VRAM đỉnh: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
   
