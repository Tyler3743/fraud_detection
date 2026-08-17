import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_node import load_data
from paths import EDGE_ATTR

DELTA_H = 6                                  # độ dài cửa sổ trượt
NS = 1_000_000_000

# Cột sinh ra trong cửa sổ, theo thứ tự ghi ra file
WIN_COLS = [
    "w_deg_out_u", "w_deg_in_u", "w_deg_out_v", "w_deg_in_v",
    "w_fan_out_u", "w_fan_in_u", "w_fan_out_v", "w_fan_in_v",
    "w_amt_out_u", "w_amt_in_v",
    "w_pair_cnt", "w_recip_cnt", "w_scatter", "w_gather",
    "w_ratio_cross_bank_u", "w_ratio_cross_ccy_u",
]
STATIC_COLS = ["is_cross_bank", "is_self_loop"]
FEAT_COLS = WIN_COLS + STATIC_COLS
# cột dạng đếm/tổng -> log1p trước khi chuẩn hoá; hai cột ratio và hai cột cờ thì không
LOG_COLS = WIN_COLS[:14]

GRAPH_SPLITS = {                             
    "train": ["train"],
    "val":   ["train", "val"],
    "test":  ["train", "val", "test"],
}


def sliding_window(t, u, v, amt, cb, cc, n_node, delta_ns, log_every=500_000):
    n = len(t)
    lo_of = np.searchsorted(t, t - delta_ns, side="left")   # biên trái, không giảm

    # đếm theo node: dùng list Python, nhanh hơn hẳn numpy khi truy cập vô hướng
    deg_out = [0] * n_node
    deg_in = [0] * n_node
    fan_out = [0] * n_node
    fan_in = [0] * n_node
    amt_out = [0.0] * n_node
    amt_in = [0.0] * n_node
    cb_out = [0.0] * n_node
    cc_out = [0.0] * n_node
    pc = {}                                                  # pair_key -> số giao dịch trong cửa sổ

    out = np.zeros((n, len(WIN_COLS)), dtype="float32")
    lo = 0
    for i in range(n):
        ui = u[i]; vi = v[i]
        k = ui * n_node + vi

        # --- nạp dòng i vào cửa sổ ---
        c = pc.get(k, 0)
        pc[k] = c + 1
        if c == 0:                                           # cặp mới -> đổi fan hai đầu
            fan_out[ui] += 1
            fan_in[vi] += 1
        deg_out[ui] += 1
        deg_in[vi] += 1
        a = amt[i]
        amt_out[ui] += a
        amt_in[vi] += a
        cb_out[ui] += cb[i]
        cc_out[ui] += cc[i]

        # --- nhả các dòng đã rơi khỏi cửa sổ ---
        li = lo_of[i]
        while lo < li:
            uo = u[lo]; vo = v[lo]
            ko = uo * n_node + vo
            co = pc[ko] - 1
            if co == 0:
                del pc[ko]
                fan_out[uo] -= 1
                fan_in[vo] -= 1
            else:
                pc[ko] = co
            deg_out[uo] -= 1
            deg_in[vo] -= 1
            ao = amt[lo]
            amt_out[uo] -= ao
            amt_in[vo] -= ao
            cb_out[uo] -= cb[lo]
            cc_out[uo] -= cc[lo]
            lo += 1

        # --- đọc đặc trưng ---
        do_u = deg_out[ui]
        rev = pc.get(vi * n_node + ui, 0)                    # chu trình độ dài 2
        fo_v = fan_out[vi]
        r = out[i]
        r[0] = do_u
        r[1] = deg_in[ui]
        r[2] = deg_out[vi]
        r[3] = deg_in[vi]
        r[4] = fan_out[ui]
        r[5] = fan_in[ui]
        r[6] = fo_v
        r[7] = fan_in[vi]
        r[8] = amt_out[ui]
        r[9] = amt_in[vi]
        r[10] = pc[k]
        r[11] = rev
        # scatter: v chuyển tiếp cho bao nhiêu đích KHÁC u (u -> v -> w)
        r[12] = fo_v - (1 if rev else 0)
        # gather: u gom từ bao nhiêu nguồn KHÁC v trước khi gửi cho v (x -> u -> v)
        r[13] = fan_in[ui] - (1 if rev else 0)
        r[14] = cb_out[ui] / do_u
        r[15] = cc_out[ui] / do_u

        if log_every and (i + 1) % log_every == 0:
            print(f"    {i + 1:>9,}/{n:,} dòng | cửa sổ đang giữ {i - lo + 1:>8,} dòng "
                  f"| {len(pc):>8,} cặp")
    return out


def build(df):
    """Trả về DataFrame [n_tx, feature] theo ĐÚNG thứ tự file gốc."""
    node = pd.factorize(pd.concat([df["src"], df["dest"]], ignore_index=True),
                        sort=False)[0]
    n = len(df)
    u = node[:n].astype("int64")
    v = node[n:].astype("int64")
    n_node = int(node.max()) + 1

    t = df["time"].to_numpy("int64")
    assert (np.diff(t) >= 0).all(), "load_data phải trả về df đã sắp theo Timestamp"

    print(f"  {n:,} giao dịch | {n_node:,} node | cửa sổ {DELTA_H}h")
    w = sliding_window(t, u, v,
                       df["Amount Paid"].to_numpy("float64"),
                       df["is_cross_bank"].to_numpy("float64"),
                       df["is_cross_curcy"].to_numpy("float64"),
                       n_node, DELTA_H * 3600 * NS)

    out = pd.DataFrame(w, columns=WIN_COLS, index=df.index)
    out["is_cross_bank"] = df["is_cross_bank"].to_numpy("float32")
    out["is_self_loop"] = (df["src"] == df["dest"]).to_numpy().astype("float32")
    out["src"] = df["src"].to_numpy()
    out["dest"] = df["dest"].to_numpy()
    out["split"] = df["split"].to_numpy()
    out["ori_idx"] = df["ori_idx"].to_numpy()
    return out.sort_values("ori_idx").reset_index(drop=True)


def verify(df, feat, n_head=1_000_000):
    banned = {"Is Laundering", "is_mule", "is_edge_mule"}
    assert not (banned & set(feat.columns)), "có cột suy từ nhãn lọt vào edge_attr"

    n_head = min(n_head, len(df) // 2)
    head = build(df.iloc[:n_head].copy()).sort_values("ori_idx")
    ref = feat.set_index("ori_idx").loc[head["ori_idx"].to_numpy(), FEAT_COLS]
    bad = ~np.isclose(head[FEAT_COLS].to_numpy("float64"),
                      ref.to_numpy("float64"), rtol=0, atol=1e-4)
    assert not bad.any(), (
        f"{int(bad.any(1).sum()):,} dòng lệch khi cắt đuôi dữ liệu; "
        f"cột đầu tiên lệch = {np.array(FEAT_COLS)[bad.any(0)][:5]}")
    print(f"  PASS: {n_head:,} dòng đầu không đổi khi bỏ toàn bộ phần sau")

    for s in ("train", "val", "test"):
        m = feat["split"] == s
        nz = (feat.loc[m, WIN_COLS].to_numpy() != 0).any(1).mean()
        print(f"  {s:5s}: {int(m.sum()):>9,} dòng | có ít nhất 1 đặc trưng khác 0: {nz*100:5.2f}%")


def scale(feat):
    feat[LOG_COLS] = np.log1p(feat[LOG_COLS].clip(lower=0))
    tr = (feat["split"] == "train").to_numpy()
    sc = StandardScaler().fit(feat.loc[tr, LOG_COLS].to_numpy("float64"))
    feat[LOG_COLS] = sc.transform(feat[LOG_COLS].to_numpy("float64")).astype("float32")
    return feat


def main():
    df = load_data()
    print("dựng đặc trưng cửa sổ trượt:")
    feat = build(df)
    print("kiểm tra:")
    verify(df, feat)
    feat = scale(feat)

    cols = ["src", "dest"] + FEAT_COLS
    for name, splits in GRAPH_SPLITS.items():
        sub = feat[feat["split"].isin(splits)]
        p = EDGE_ATTR.format(name)
        sub[cols].to_parquet(p, index=False)
        print(f"  đã lưu {p} shape={sub[cols].shape} "
              f"({sub[cols].memory_usage(deep=False).sum()/1e6:.0f} MB trong RAM)")
        del sub; gc.collect()


if __name__ == "__main__":
    main()
