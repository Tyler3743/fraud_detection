import json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import MessagePassing

from metrics import find_best_threshold, evaluate, log_result

DATA = "dataset_high"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]
HID, NEG_RATIO, EPOCHS, BATCH, LR, PATIENCE = 32, 50, 200, 32768, 3e-3, 20 #cần tune lại


class EdgeSAGE(MessagePassing):
    """SAGE có edge_attr: message = MLP([h_j ‖ edge_attr]), aggr=mean, + nhánh root."""
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr="mean")
        self.msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.root = nn.Linear(in_dim, out_dim)

    def forward(self, h, edge_index, edge_attr):
        return self.root(h) + self.propagate(edge_index, x=h, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return F.relu(self.msg(torch.cat([x_j, edge_attr], dim=-1)))


class Model(nn.Module):
    def __init__(self, num_banks, edge_dim, tx_dim, hid=HID):
        super().__init__()
        self.bank = nn.Embedding(num_banks, hid)
        self.c1 = EdgeSAGE(hid, edge_dim, hid)
        self.c2 = EdgeSAGE(hid, edge_dim, hid)
        self.head = nn.Sequential(
            nn.Linear(2 * hid + tx_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def encode(self, g):
        h = self.bank(g.x.squeeze(-1))
        h = F.relu(self.c1(h, g.edge_index, g.edge_attr))
        return self.c2(h, g.edge_index, g.edge_attr)

    def forward(self, h, pairs, tx):
        return self.head(torch.cat([h[pairs[:, 0]], h[pairs[:, 1]], tx], -1)).squeeze(-1)


def load_all(no_edge_attr=False):
    split = pd.read_parquet(f"{DATA}/transaction_features.parquet", columns=["split"])["split"].to_numpy()
    nodes = np.load(f"{DATA}/txn_nodes.npy")
    blob = torch.load(f"{DATA}/graphs.pt", weights_only=False)
    graphs, num_banks = blob["graphs"], blob["num_banks"]

    d = {}
    for s in ["train", "val", "test"]:
        m = pd.read_parquet(f"{DATA}/txn_matrix_{s}.parquet")
        y = m.pop("Is Laundering").to_numpy().astype("float32")
        d[s] = [m.to_numpy("float32"), y, torch.from_numpy(nodes[split == s])]

    sc = StandardScaler().fit(d["train"][0])                 # fit CHỈ trên train
    for s in d:
        d[s][0] = sc.transform(d[s][0]).astype("float32")

    for s, g in graphs.items():
        if no_edge_attr:
            g.edge_attr = torch.zeros_like(g.edge_attr)      # đối chứng topology-only
        graphs[s] = g.to(DEV)
    return d, graphs, num_banks


def sample_epoch(y, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
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


def run(seed, d, graphs, num_banks, tag):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr, ytr, ptr = d["train"]
    model = Model(num_banks, graphs["train"].edge_attr.size(1), Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    amp = DEV.type == "cuda"
    gscaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(EPOCHS):
        idx = sample_epoch(ytr, rng)
        model.train()
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                h = model.encode(graphs["train"])            # full-graph: 515K node, đủ nhỏ
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

    s_val, h_val = score(model, graphs["val"], d["val"][0], d["val"][2])
    s_test, h_test = score(model, graphs["test"], d["test"][0], d["test"][2])
    thr = find_best_threshold(d["val"][1], s_val)
    for s, sc_ in [("val", s_val), ("test", s_test)]:
        log_result(tag, s, evaluate(d[s][1], sc_, thr), seed=seed, stage="final",
                   train_time_s=train_time_s,
                   params=json.dumps({"hid": HID, "neg_ratio": NEG_RATIO, "epochs_run": ep + 1}))
        np.save(f"scores/{tag}_seed{seed}_{s}.npy", sc_)

    # embedding cho nhóm 3 (graph lũy tiến, không dùng nhãn test)
    np.save(f"scores/emb_{tag}_seed{seed}_val.npy", h_val.float().cpu().numpy())
    np.save(f"scores/emb_{tag}_seed{seed}_test.npy", h_test.float().cpu().numpy())
    print(f"[{tag} seed{seed}] {train_time_s}s | val_pr_auc={best:.4f}")


def main(mode="edge"):
    os.makedirs("scores", exist_ok=True)
    tag = "sage" if mode == "edge" else "sage_noedge"
    d, graphs, num_banks = load_all(no_edge_attr=(mode != "edge"))
    for seed in SEEDS:
        run(seed, d, graphs, num_banks, tag)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "edge")