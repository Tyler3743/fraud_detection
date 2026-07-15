# plan.md — Phát hiện giao dịch rửa tiền: so sánh 3 nhóm model ở cấp giao dịch

File context. Đọc đầu mỗi phiên để nắm trạng thái + kế hoạch.

---

## 1. Tổng quan

**Đề tài (tạm):** Phát hiện gian lận giao dịch tài chính kết hợp GNN và XGBoost.

**Hướng làm:** GNN mã hóa ở **cấp tài khoản** (nhẹ, ~515K node) nhưng **phân loại ở cấp giao dịch** (`Is Laundering`) để so trực tiếp với literature (Altman/GFP/Egressy).

**Khung thực nghiệm (đã chốt): 3 nhóm model, TẤT CẢ đánh giá ở cấp giao dịch, cùng split, cùng minority-F1.** Khác nhau ở mức độ dùng đồ thị:

- **Nhóm 1 — classical:** LR, DT, RF, XGBoost, MLP. Không message passing; phân loại từng giao dịch bằng transaction feature (+ node feature 2 đầu).
- **Nhóm 2 — GNN:** GCN, GraphSAGE, GAT, GIN, skip-GCN (BWGNN, evolveGCN optional). Encoder trên graph tài khoản gộp cạnh + head phân loại từng giao dịch (end-to-end).
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
│ ├─ HI-Small_Trans_split_index.csv # đã gán cột 'split' (60/20/20)
│ ├─ node_features*{train,val,test}.csv # feature_node.py ✅
│ ├─ edge_features\*{train,val,test}.csv # feature_edge.py ✅
│ └─ transaction_features.parquet # feature_transaction.py ✅
├─ dataset_small/ # LI-Small (cross-test)
├─ AccountFraudLabel.csv # nhãn node-mule (optional aux)
├─ nodelabel.py / temporal_split_index.py / feature_node.py
├─ feature_edge.py / feature_transaction.py  
├─ metrics.py (➕) độ đo chung cấp giao dịch
├─ assemble_txn.py (➕) ghép ma trận per-transaction cho nhóm 1 & 3
├─ build_graph.py (➕) dựng + cache PyG graph, map node→idx, tx→(src,dst)
├─ train_classical.py (➕) nhóm 1
├─ train_gnn.py (➕) nhóm 2 (models inline)
├─ train_hybrid.py (➕) nhóm 3 + ablation
├─ baseline_gfp.py (➕) GFP + XGBoost (đối thủ)
└─ results.csv (➕ auto) log mọi run

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

- Cắt theo **index thời gian 60/20/20** (chuẩn Altman), 2 mốc t1,t2; assert no-leakage.
- **Graph lũy tiến:** train-graph = train edges; val-graph = train+val; test-graph = all; chỉ eval trên index tương ứng.
- Mọi feature chỉ học từ cửa sổ cho phép; scaler fit **chỉ trên train**.

---

## 5. Thiết kế kỹ thuật — thực nghiệm 3 nhóm (unified, cấp giao dịch)

### Bước 0 — Nhãn

Nhãn = `Is Laundering` theo mỗi giao dịch (nhãn thật, so được literature). Nhãn node-mule (`AccountFraudLabel.csv`, quy tắc: account dương nếu dính ≥1 giao dịch laundering) chỉ là tùy chọn (cho ablation V1 + diễn giải).

### Bước 1 — Danh tính node (tuple)

Khóa `(Bank, Account)`: gửi `(From Bank, Account)`, nhận `(To Bank, Account.1)`. Bắt buộc vì có account trùng mã giữa bank (đã xác nhận `check_duplicate.ipynb`). Đọc string.

### Bước 2 — Split

Đã làm (mục 4).

### Bước 3 — Feature engineering (3 loại, cửa sổ lũy tiến, scaler train-only)

- **Node feature** (`feature_node.py` ✅, ~26 cột/account): dòng tiền, `net_flow`, degree in/out, xuyên bank, currency mix, payment format, velocity, số tròn. → làm `x` cho GNN (nhóm 2,3) + join 2 đầu cho nhóm 1.
- **Transaction feature** (`feature_transaction.py` ✅, giao dịch): amount log, cross-bank/currency, round, self-loop, giờ (sin/cos), one-hot payment format + currency. → input **chính** nhóm 1; ghép nối nhóm 2; vào XGBoost tầng 2 nhóm 3.
- **Edge feature** (`feature_edge.py` ✅, tính feature cho từng edge): thống kê tiền/đếm, tỉ lệ cross-ccy/round, active_day. thông tin của transaction cho GNN (nhóm 2,3)
- **KHÔNG** nhồi pattern-count kiểu GFP vào feature GNN — để GNN tự học, giữ đóng góp 1 sạch; pattern để dành baseline GFP.
- ⚠️ **Leakage:** khi ghép dữ liệu feature vào giao dịch phải **bỏ cột `is_edge_mule`** (suy từ nhãn).

### Bước 4 — Lắp dữ liệu & dựng graph

- `assemble_txn.py`: mỗi giao dịch ⟵ `[transaction feat] + [node feat src] + [node feat dest] (+ edge feat cặp)`. → ma trận nhóm 1 & 3.
- `build_graph.py`: PyG graph lũy tiến — `x`=node feat, `edge_index`+`edge_attr`=edge feat (gộp cạnh, đối xứng). Cache `.pt` (dựng 1 lần). Kèm map `node_key→idx` và `transaction→(src_idx,dst_idx)` cho readout.
- 1 notebook duy nhất (vd analysis.ipynb) — chỉ đọc results.csv để tính mean ± std, vẽ bảng/biểu đồ cho luận văn. Không train gì trong notebook.

### Bước 5 — Nhóm 1 (classical, per-transaction) — **sàn**

Chạy trước (nhanh, không graph). LR/DT/RF/XGBoost/MLP trên ma trận `assemble_txn`. **XGBoost tx-only = V0** (baseline "XGBoost thuần"). Trả lời: feature thủ công + cây đạt bao nhiêu, graph có hơn không. đầu ra bao gồm:

- results.csv — mỗi dòng = (model, seed, split): f1_minority, precision, recall, pr_auc, recall@fpr1%, precision@1000, threshold, seed, train_time_s. Chạy 5 seed/model → notebook phân tích tính mean ± std. (metrics.py đã có log_result, chỉ cần thêm seed=, train_time_s= qua \*\*extra.)
- Điểm dự đoán thô — lưu scores/{model}_seed{s}_{split}.npy (y_score của val + test). Có cái này thì vẽ lại PR curve, confusion matrix, đổi threshold... cho luận văn mà không phải train lại.
- Best hyperparameters — ghi vào cột extra trong results.csv (dạng chuỗi JSON) → làm bảng phụ lục giống Altman.
  (Nên có) train_time_s mỗi model — bằng chứng chi phí cho đóng góp 1.
- (Tùy chọn) model XGBoost đã train (.json) + feature importance — dùng cho phần SHAP/diễn giải bước 8, và V0 tái dùng ở nhóm 3 khỏi train lại.

### Bước 6 — Nhóm 2 (GNN + edge head, end-to-end)

- Encoder message passing trên graph tài khoản (2 layer, neighbor sampling `[15,10]`, batch 512, AMP). Head chấm điểm từng giao dịch: `head([emb_src ‖ emb_dst ‖ tx_feat])`, BCE per-transaction + `pos_weight`.
- Screen họ message-passing 1 cấu hình → tune top 1–2 → **chọn encoder tốt nhất**. (BWGNN/evolveGCN optional, không khớp harness chung.)
- Xuất node embedding (+ điểm mule nếu bật aux head) cho nhóm 3.

### Bước 7 — Nhóm 3 (hybrid) + ablation + đối thủ

- Vector giao dịch = `[tx feat] + [emb src] + [emb dest] (+ điểm mule 2 đầu)` → **XGBoost (CPU)** học `Is Laundering`.
- **Ablation:** V0 (tx-only, từ Bước 5) → V2 (tx + embedding). V1 (+ điểm mule) optional nếu bật aux head. Delta V0→V2 = bằng chứng đóng góp 1.
- **Đối thủ thật:** `baseline_gfp.py` — GFP (Snap ML) + XGBoost, chạy cùng máy.
- Embedding cho giao dịch test tính theo **graph lũy tiến chuẩn Altman** (full-graph, cấm dùng label test).

---

## 6. Độ đo & chống leakage

- **Metric (chung, `metrics.py`):** minority-F1 cấp giao dịch (chính, so 63.23/68.16), PR-AUC, recall@FPR, precision@k. Threshold tune trên val. **Ghi rõ split** mọi kết quả.
- hàm mất mát: weightes-cross entropy, focal loss
- **Mất cân bằng (cross-cutting):** `class_weight` (LR/RF), `scale_pos_weight` (XGBoost), `pos_weight`/focal (GNN).
- **Chống leakage:** scaler fit train-only; graph lũy tiến; bỏ `e_is_edge_mule` khi join; embedding test không dùng label test.

---

## 7. Thứ tự công việc

1. Split ✅ 2. Node feature ✅ 3. Transaction feature ✅ 4. Edge feature ✅
2. `metrics.py` chung.
3. `assemble_txn.py` → **Nhóm 1** (`train_classical.py`): sàn + V0. notebook analysis.ipynb
4. `build_graph.py` (cache graph).
5. **Nhóm 2** (`train_gnn.py`): screen → tune → chọn encoder.
6. Xuất embedding (+ điểm mule optional).
7. **Nhóm 3** (`train_hybrid.py`): XGBoost + ablation V0/V2 + `baseline_gfp.py`.
8. Chọn pipeline tốt nhất; (optional) grouped SHAP minh họa.
9. (Sau) cross-test HI→LI; rồi mới sang đóng góp 2 (real-time).

---

## 8. Định vị khoa học & đóng góp

**Baseline HI-Small (minority-F1, đã xác minh):** GIN 28.70 · GIN+EU 47.73 · PNA 56.77 · GFP+LightGBM 62.86 · GFP+XGBoost **63.23** · Multi-PNA+EU **68.16** (Egressy, mốc trên).

**Đóng góp 1 (chính):** encoder GNN nhẹ (cấp tài khoản, ~515K node) + XGBoost đạt bao nhiêu % hiệu năng của GNN edge-level nặng (~5M cạnh, Egressy) với chi phí thấp hơn bao nhiêu? Bằng chứng = **ablation V0→V2** trên split-index. Đối thủ chạy lại được: GFP+XGBoost; mốc citation: Egressy 68.16.

**Đóng góp 2 (làm sau, đo systems):** phát hiện real-time 2 đường (đường nóng XGBoost/CPU đạt SLA p99; đường nguội GNN inductive refresh embedding/GPU) — trục đo latency/throughput + đường cong staleness. Làm chắc đóng góp 1 trước.

---

## 9. Trade-off & vấn đề mở

- **Gộp cạnh + readout per-tx** (nhẹ, hợp 4GB) vs **multigraph mỗi giao dịch = 1 cạnh** (đúng Egressy, nặng): chọn gộp cạnh → số GNN nhóm 2 **không phải reproduce GIN 28.7 của Altman** mà là phương pháp đề xuất; muốn baseline GNN so cứng thì chỉ trích số công bố.
- **Nhóm 2 end-to-end trên 5M nhãn cạnh** là phần tốn compute nhất — nút thắt thời gian.
- **Screen nhanh rồi tune top 1–2** thay vì HPO đủ 12+ model: tiết kiệm mạnh, đổi lại có thể bỏ sót model thắng nếu tune kỹ.
- **BWGNN/evolveGCN optional:** không khớp harness chung (spectral full-batch / cần snapshot ngày); evolveGCN hợp đóng góp 2 hơn.
- **Điểm mule (V1) optional:** bật aux head thì rẻ + có V1 + diễn giải; bỏ thì ablation gọn V0→V2.
- **Edge feature nhãn khác cấp:** `e_is_edge_mule` là nhãn cạnh-gộp, chỉ join làm feature (đã bỏ khi train), không dùng làm mục tiêu eval.
- **Vấn đề mở:** label leakage nhẹ (nhãn node global timeline — v1 chấp nhận); tie-breaking split khi tái dùng cho LI-Small cần kiểm edge case;
