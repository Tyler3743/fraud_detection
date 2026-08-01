# plan.md — Phát hiện giao dịch rửa tiền: so sánh 3 nhóm model ở cấp giao dịch

File context. Đọc đầu mỗi phiên để nắm trạng thái + kế hoạch.

---

## 1. Tổng quan

**Đề tài (tạm):** Phát hiện gian lận giao dịch tài chính kết hợp GNN và XGBoost.

**Hướng làm:** GNN mã hóa ở **cấp tài khoản** (nhẹ, ~515K node) nhưng **phân loại ở cấp giao dịch** (`Is Laundering`) để so trực tiếp với literature (Realistic synthetic money, Graph Feature Preprocessor, Provably powerful graph network, Extracting money laundering, money laundering detection with multi-GIN-những paper này đều có dạng markdown trong folder IBM AML).

**Khung thực nghiệm (đã chốt): 3 nhóm model, TẤT CẢ đánh giá ở cấp giao dịch, cùng split, cùng minority-F1.** Khác nhau ở mức độ dùng đồ thị:

- **Nhóm 1 — classical:** LR, DT, RF, XGBoost, MLP. Không message passing; phân loại từng giao dịch bằng transaction feature và as-of aggregate (tổng hợp feature không làm temporal leaky)
- **Nhóm 2 — GNN:** GCN, GraphSAGE, GAT, GIN. Encoder trên graph tài khoản gộp cạnh (edge_attr lagged) + head phân loại từng giao dịch (end-to-end).
- **Nhóm 3 — hybrid:** encoder nhóm 2 + **XGBoost** làm head. Đây là pipeline đề xuất (đóng góp 1).

GraphSAGE chỉ là **một ứng viên**, kiến trúc có thể đổi.

**Dataset:** IBM AMLworld — train/eval HI-Small (`dataset_high/`), LI-Small (`dataset_small/`) để cross-test HI→LI.

**Ràng buộc phần cứng:** RAM 16GB, RTX 2050 VRAM 4GB → mọi quyết định (gộp cạnh, neighbor sampling, mixed precision) xoay quanh đây.

---

## 2. Cấu trúc thư mục & file

D:\ct551*v2
├─ dataset_high/ # HI-Small
│ ├─ HI-Small_Trans.csv # ~5.08M giao dịch, nhãn 'Is Laundering'
│ ├─ HI-Small_accounts.csv
│ ├─ HI-Small_Trans_split_index.csv # cột 'split' (60/20/20) ✅
│ ├─ transaction_features.parquet # feature_transaction.py ✅ (causal, giữ nguyên)
│ ├─ node_edge_features.parquet # feature_node.py ➕ (as-of per transaction)
│ ├─ edge_attr*{train,val,test}.csv # feature_edge.py ♻️ viết lại (lagged)
├─ dataset_small/ # LI-Small (cross-test)
├─ AccountFraudLabel.csv # nhãn node-mule (optional aux V1)
├─ nodelabel.py / temporal_split_index.py
├─ feature_transaction.py ✅ / feature_node.py ♻️ / feature_edge.py ♻️
├─ metrics.py ✅
├─ assemble_txn.py ♻️ # ghép tx + as-of theo index (không join CSV)
├─ build_graph.py ➕ # PyG graph lũy tiến + edge_attr lagged, cache .pt
├─ train_classical.py ✅ (rerun) # nhóm 1
├─ train_gnn.py ➕ # nhóm 2
├─ train_hybrid.py ➕ # nhóm 3 + ablation
├─ baseline_gfp.py ➕ # GFP + XGBoost (đối thủ)
├─ results.csv (➕ auto) # log mọi run SAU fix leakage
└─ results-leaky.csv # kết quả trước fix — giữ làm bảng before/after

**Cột Trans.csv:** `Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`.
**Leading zero:** đọc bank/account dạng **string** khi join.

---

## 3. Số liệu dataset (đã xác minh, HI-Small)

- Tổng giao dịch: **5,078,345**; ~99% nằm ngày 1–10 (2022/09).
- Laundering cấp giao dịch: **5,177 ≈ 0.10%** (mất cân bằng cực mạnh — đây là nhãn eval).
- Node tài khoản: **515,088**, dương **6,357 ≈ 1.23%** (dùng cho aux node-mule + graph).
- Distribution shift: fraud rate tăng train→test (~2.2×) → dùng PR-AUC/Recall, không dùng Accuracy.

---

## 4. Split temporal (đã chốt)

- Cắt theo **index thời gian 60/20/20**
- **Graph lũy tiến (cấu trúc):** train-graph = train edges; val-graph = train+val, test-graph = all; chỉ eval trên index tương ứng. Cấu trúc được phép lũy tiến, chuẩn inductive Altman, mọi baseline chấp nhận.
- Mọi feature phải tuân theo quy luật nhân quả (causal); scaler fit **chỉ trên train**.

---

## 5. Thiết kế kỹ thuật — thực nghiệm 3 nhóm (unified, cấp giao dịch)

### Bước 0 — Nhãn

Nhãn = `Is Laundering` theo giao dịch. Nhãn node-mule (`AccountFraudLabel.csv`) chỉ
tùy chọn (aux V1 + diễn giải), không bao giờ làm feature.

### Bước 1 — Danh tính node (tuple)

Khóa `(Bank, Account)`: gửi `(From Bank, Account)`, nhận `(To Bank, Account.1)`.
Đọc string (account trùng mã giữa bank — check_duplicate.ipynb).

### Bước 2 — Split

Đã làm (mục 4).

### Bước 3 — Feature engineering (causal, scaler train-only)

Graph có hai vai: **cấu trúc** (được lũy tiến) và **feature số** (phải causal/lagged).

- **Transaction feature** (`feature_transaction.py` ✅): amount log, cross-bank/currency,
  round, self-loop, giờ/dow sin-cos, one-hot format+currency. Causal tự nhiên (chỉ mô
  tả chính giao dịch). → input chính nhóm 1; ghép head nhóm 2; vào XGBoost nhóm 3.
- **As-of aggregate** (`feature_node.py` ➕, cho nhóm 1 & 3): mỗi giao dịch t nhận
  thống kê tích lũy của src / dest / cặp (src,dest) từ các giao dịch TRƯỚC t
  (sort Timestamp → groupby + cumsum/cummax + shift(1)). Cột: count/sum/mean/std/
  min/max amount (2 vai gửi–nhận), nunique_prior (đối tác, bank, currency — qua cờ
  flag), tỉ lệ cross-bank/ccy/round, tỉ lệ payment format, active time, tx_per_day, cờ first_seen (cold-start fill 0). Bỏ median/skew/kurt vì GFP xác nhận O(Δ), không streaming rẻ.
  Bỏ skew/kurtosis vì trong chế độ as-of/cold-start chúng phần lớn không định nghĩa được (cần ≥3–4 giao dịch lịch sử, cell thứ 3 trong skew_edge.ipynb đã minh chứng điều này) và moment bậc cao bất ổn số học khi tính tích lũy trong pandas — dù GFP tính được O(1) bằng central moment streaming trong C++.
- **Edge_attr cho GNN** (`feature_edge.py` ♻️, **lagged causal**): thống kê cặp
  (src,dest) gộp cạnh, tính từ cửa sổ TRƯỚC cửa sổ eval:
  - train-graph ← chính cửa sổ train (không có gì trước nó — limitation, ghi §9);
  - val-graph ← train; test-graph ← train+val;
  - cạnh chưa xuất hiện trong cửa sổ nguồn → vector 0 + cờ `seen_before`.
    Cột: num_tx, total/mean/std/min/max paid, cross_ccy_ratio, round_ratio,
    is_cross_bank, is_self_loop, active_day, tx_per_day, tỉ lệ format/currency.
    BỎ `is_edge_mule`. Định vị: chặt hơn ExSTraQt (batch whole-window) — trích làm
    tiền lệ literature chấp nhận aggregate trên cạnh gộp.
- **Node `x` cho GNN:** chỉ sử dụng các đặc trưng tĩnh như Bank ID và Account ID — embedding bank id(+ hằng số). KHÔNG thống kê các đặc trưng động (tương tự Egressy/FraudGT: đặc trưng dồn vào edge).
- KHÔNG cho học các mẫu pattern giống GFP vào feature GNN — giữ đóng góp 1 sạch.
- ⚠️ `is_mule` / `is_edge_mule`: suy từ nhãn — không bao giờ join làm feature.

### Bước 4 — Lắp dữ liệu & dựng graph

- `assemble_txn.py` ♻️: ma trận = `[tx feat + as-of src + as-of dest + as-of pair]`,
  ghép **cùng index giao dịch** (không join CSV theo split); assert NaN/hit-rate.
- `build_graph.py` ➕: PyG graph lũy tiến — `x` = bank-id index (embedding trong model),
  `edge_index` gộp cạnh đối xứng, `edge_attr` lagged (Bước 3). Cache `.pt`.
  Kèm map `node_key→idx`, `transaction→(src_idx,dst_idx)` cho readout.
- 1 notebook duy nhất (analysis.ipynb) — chỉ đọc results.csv, không train.

### Bước 5 — Nhóm 1 (classical, per-transaction) — **sàn**

LR/DT/RF/XGBoost/MLP trên ma trận assemble mới. **XGBoost = V0**. Đầu ra: results.csv
(model, seed, split, f1_minority, precision, recall, pr_auc, recall@fpr1%,
precision@1000, threshold, train_time_s, params JSON), scores, npy, model V0 (.json)

- feature importance. 5 seed/model. ⚠️ Kết quả cũ (leaky) đã rename results-leaky.csv
- giữ làm bảng before/after cho luận văn.
- các bài báo có dùng node feature liên quan (trong folder md IBM AML): Anti-Money Laundering Alert Optimization, Application of Classical & Quantum-Hybrid, Privacy-Preserving Graph-Based, Amatriciana (Temporal GNN), Graph Feature Preprocessor, NETWORK ANALYTICS trong folder md IBM AML , Provably Powerful Multigraph (Egressy), fraudGT, Realistic Synthetic

### Bước 6 — Nhóm 2 (GNN + edge head, end-to-end)

- Encoder message passing trên graph Bước 4 (2 layer, neighbor sampling [15,10],
  batch 512, AMP). **Encoder phải nhận edge_attr**: GIN, GAT(edge_dim),
  SAGE+edge-concat; GCN thuần = biến thể không edge_attr (đối chứng topology-only).
- Head: `head([emb_src ‖ emb_dst ‖ tx_feat])`, BCE per-transaction + pos_weight/focal.
- Screen 1 cấu hình/họ → tune top 1–2 → chọn encoder tốt nhất.
- Xuất node embedding (+ điểm mule nếu bật aux) cho nhóm 3.

### Bước 7 — Nhóm 3 (hybrid) + ablation + đối thủ

- Vector = `[V0 feat + emb_src + emb_dest] (+ điểm mule)` → **XGBoost (CPU)**.
- **Ablation:** V0 (Bước 5) → V2 (V0 + embedding). Delta V0→V2 = phần embedding đóng
  góp THÊM trên sàn mạnh = bằng chứng đóng góp 1. V1 (+mule) optional.
- **Đối thủ thật:** `baseline_gfp.py` — GFP (Snap ML) + XGBoost, cùng máy, cùng split.
- Embedding test tính theo graph lũy tiến chuẩn Altman (cấm dùng label test).

---

## 6. Độ đo & chống leakage

- **Metric (`metrics.py`):** minority-F1 cấp giao dịch (chính, so 63.23/68.16),
  PR-AUC, recall@FPR, precision@k. Threshold tune trên val. Ghi rõ split.
- Hàm mất mát: weighted cross-entropy, focal loss.
- **Mất cân bằng:** class_weight (LR/RF), scale_pos_weight (XGBoost), pos_weight/focal (GNN).
- **Chống leakage (đã chốt):**
  - Feature số nhóm 1/3: as-of per transaction — không thống kê nào thấy t' ≥ t.
  - edge_attr GNN: lagged theo cửa sổ (val←train, test←train+val).
  - Cấu trúc graph: lũy tiến chuẩn Altman (được phép).
  - Scaler fit train-only; embedding test không dùng label test;
    is_mule/is_edge_mule không làm feature.

---

## 7. Thứ tự công việc

1. Split ✅ 2. Transaction feature ✅ 3. metrics.py ✅
2. `feature_node.py` ➕ → verify: (a) tx đầu tiên của account có aggregate=0,
   first_seen=0; (b) count as-of cuối cùng của account == count whole-window;
   (c) spot-check 1 tx test: tính tay từ dữ liệu < t, so khớp.
3. `assemble_txn.py` ♻️ → **Nhóm 1 rerun** (`train_classical.py`) → so bảng leaky/sạch.
4. `feature_edge.py` ♻️ (lagged) + `build_graph.py` → verify: edge_attr val không đổi
   khi xáo dữ liệu val (chỉ phụ thuộc train).
5. **Nhóm 2** (`train_gnn.py`): screen → tune → chọn encoder.
6. Xuất embedding (+ mule optional).
7. **Nhóm 3** (`train_hybrid.py`): XGBoost + ablation V0/V2 + `baseline_gfp.py`.
8. Chọn pipeline tốt nhất; (optional) grouped SHAP.
9. (Sau) cross-test HI→LI; rồi đóng góp 2 (real-time).

---

## 8. Định vị khoa học & đóng góp

**Baseline HI-Small (minority-F1, đã xác minh):** GIN 28.70 · GIN+EU 47.73 · PNA 56.77
· GFP+LightGBM 62.86 · GFP+XGBoost **63.23** · Multi-PNA+EU **68.16** (Egressy, mốc trên).

Bài đo được baseline khác (folder IBM AML): Realistic Synthetic (Altman), **ExSTraQt**
(Extracting Money Laundering — gộp cạnh, batch whole-window aggregate → tiền lệ cho
edge_attr trên cạnh gộp; phương án của ta lagged, chặt hơn), Graph Feature
Preprocessor, MAGIC, Finding Money Launderers (Jensen).

**Đóng góp 1 (chính):** encoder GNN nhẹ (cấp tài khoản) + XGBoost đạt bao nhiêu %
hiệu năng GNN edge-level nặng (Egressy) với chi phí thấp hơn bao nhiêu? Bằng chứng =
ablation V0→V2, feature causal cả hai phía. Đối thủ chạy lại: GFP+XGBoost (cùng ngữ
nghĩa causal streaming → head-to-head hợp lệ); mốc citation: Egressy 68.16, FraudGT 68.6.

**Đóng góp 2 (sau, systems):** real-time 2 đường (nóng XGBoost/CPU, nguội GNN refresh
embedding/GPU) — latency/throughput + staleness. Làm chắc đóng góp 1 trước.

---

## 9. Trade-off & vấn đề mở

- **Temporal leakage ĐÃ xử lý (v2):** as-of feature (nhóm 1/3) + edge_attr lagged (GNN).
  results-leaky.csv giữ làm bằng chứng before/after.
- **Limitation phải khai:** (a) edge_attr train-graph dùng chính cửa sổ train (không có
  dữ liệu trước) → mismatch ngữ nghĩa train/test — cùng tinh thần ExSTraQt batch;
  (b) cold-start: cạnh/account mới trong test có aggregate=0 — đúng vùng fraud hay nằm,
  cờ seen_before giúp model học điều này.
- **Gộp cạnh + trích xuất feature per-tx** vs multigraph Egressy: nhóm 2 KHÔNG reproduce GIN 28.7
  — là phương pháp đề xuất; baseline GNN so cứng thì trích số công bố.
- **Nhóm 2 end-to-end 5M nhãn** = nút thắt compute.
- **thử nghiệm các mô hình khác nhau rồi tune top 1–2** — có thể bỏ sót model thắng.
- **BWGNN/evolveGCN optional** (không khớp harness; evolveGCN hợp đóng góp 2).
- **Điểm mule (V1) optional**; `e_is_edge_mule` không bao giờ làm feature.
- **Vấn đề mở:** nhãn node-mule dòng thời gian toàn cục (chỉ dự phòng, v1 chấp nhận); tie-breaking
  split khi tái dùng LI-Small cần kiểm edge case.
