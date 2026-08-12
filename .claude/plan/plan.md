# plan.md — Phát hiện giao dịch rửa tiền: so sánh 3 nhóm model ở cấp giao dịch

File context. Đọc đầu mỗi phiên để nắm trạng thái + kế hoạch.

---

## 1. Tổng quan

**Đề tài (tạm):** Hệ thống phát hiện gian lận giao dịch tài chính thời gian thực (có thể sử dụng classifier hoặc graphsage+classfier)

**Hướng làm:** GNN mã hóa ở **cấp tài khoản** (nhẹ, ~515K node) nhưng **phân loại ở cấp giao dịch** (`Is Laundering`) để so trực tiếp với literature (Realistic synthetic money, Graph Feature Preprocessor, Provably powerful graph network, Amatriciana-những paper này đều có dạng markdown trong folder IBM AML paper/markdown).

**Khung thực nghiệm (đã chốt): 3 nhóm model, TẤT CẢ đánh giá ở cấp giao dịch, cùng split, cùng minority-F1.** Khác nhau ở mức độ dùng đồ thị:

- **Nhóm 1 — classical:** LR,DT, RF,XGBoost,lightGBM. Không message passing; phân loại từng giao dịch bằng transaction feature và as-of aggregate (tổng hợp feature không làm temporal leaky) nhóm 1 tune trong analysis_1.ipynb
- **Nhóm 2 — GNN:** GraphSAGE. Encoder trên graph tài khoản gộp cạnh (edge_attr lagged) + head phân loại từng giao dịch (end-to-end). bắt buộc làm graphsage những mô hình khác chỉ tham khảo
- **Nhóm 3 — hybrid:** encoder nhóm 2 + **classifier(XGBoost,LightGBM)** làm head. Đây là pipeline đề xuất (đóng góp 1).

cách làm graphsage kết hợp LSTM tương đồng mô hình amatriciana trong bài báo amatriciana.md

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
├─ assemble_txn.py ✅
├─ build_graph.py ✅
├─ train_classical.py ✅
├─ train_gnn.py ✅
├─ train_hybrid.py ➕ # nhóm 3 + ablation
├─ results.csv chỉ giữ bản mới nhất (ghi đè không append)

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
  flag), tỉ lệ cross-bank/ccy/round, tỉ lệ payment format, active time, tx_per_day, cờ first_seen (cold-start fill 0). Bỏ median/skew/kurt vì GFP xác nhận O(Δ), không streaming rẻ. bỏ is_round
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
  `edge_index` gộp cạnh đối xứng, `edge_attr` lagged (Bước 3).
- analysis_1.ipynb

### Bước 5 — Nhóm 1 (classical, per-transaction) — **sàn**

xgboost/random forest/ lightGBM/ decision tree/ lr trên ma trận assemble mới. **classifier = V0**. Đầu ra: results.csv
(model, seed, split, f1_minority, f1@0.5, precision, recall, pr_auc, recall@fpr1%,
precision@1000, threshold, train_time_s, params JSON), scores, npy, model V0 (.json)

- kết quả lighGBM>XGBoost>Random Forest>Decision Tree>Logistic regression

### Bước 6 — Nhóm 2 (GNN + edge head, end-to-end)

- Encoder message passing trên graph Bước 4 (2 layer, neighbor sampling [15,10],
  batch 512, AMP). **Encoder phải nhận edge_attr**:
  SAGE+edge-concat = biến thể không edge_attr (đối chứng topology-only).
- thực nghiệm 4 dạng graphsage: noedge, edge, noedge+lSTM, edge+LSTM sẽ chọn ra model thắng
  sau 2 lần chạy, sau 2 lần chạy lại so sánh model thắng mới bước vào nhóm 3
- phải tuân thủ không leak dữ liệu và đồng bộ logic với đóng góp 1 mục 8
- thực hiện trên cấu hình GPU T4 x 2 (kaggle, colab)
- đã thực hiện xong nhóm 2 kết quả: train-gnn.ipynb, kết quả: results-gnn-news, variant_board, zip scores trong folder scores

### Bước 7 — Nhóm 3 hybrid: GraphSAGE(LSTM)+XGBoost/LightGBM

- trong thời gian chạy nhóm 3 dựng khung web chạy thử nghiệm với mô hình nhóm 1
- khung web gồm có các tính năng sau: có tùy chọn từ ngày nào đến ngày nào, tùy chọn mô hình, rồi
  vẽ ra biểu đồ thống kê giao dịch, thống kê feature của 1 node trong khoảng thời gian, tài khoản có
  hoạt động như thế nào (bảng tra cứu embedding vào web), lọc ra được các giao dịch gian lận và không gian lận và nút chi tiết khi bấm vào sẽ hiện ra pattern và SHAP

Quy trình đo latency:

- Tách hai chế độ, KHÔNG gộp:
- Chế độ duyệt: đọc score/feature đã precompute → UI mượt khi kéo slider.
  Đây là phần demo cho hội đồng xem.
- Chế độ đo (benchmark_latency.py): replay tuần tự theo Timestamp, tính
  as-of state LIVE, chấm điểm từng giao dịch. Chạy một lần, xuất bảng số.

  Ràng buộc khi đo:

- batch size = 1 (predict cả tập rồi chia = đo throughput theo lô, không phải latency)
- dùng numpy array, KHÔNG dựng DataFrame 1 dòng mỗi giao dịch (overhead pandas
  sẽ lớn hơn cả thời gian predict và chiếm trọn con số)
- warm-up: bỏ ~5.000 giao dịch đầu trước khi ghi số
- tách thành phần: cập nhật as-of / dựng vector / tra embedding (nhóm 3) / predict
- báo cáo p50/p95/p99 + throughput, đơn luồng
- đối chiếu ngưỡng nghiệp vụ CÓ TRÍCH DẪN (vd. độ trễ luồng authorization thẻ),
  không tự đặt ngưỡng

---

## 6. Độ đo & chống leakage

- **Metric (`metrics.py`):** minority-F1 cấp giao dịch (chính),
  PR-AUC, recall@FPR, precision@k. Threshold tune trên val. Ghi rõ split.
- Hàm mất mát:
- **Mất cân bằng:** class_weight (LR/RF), scale_pos_weight (XGBoost), (GNN).
- **Chống leakage (đã chốt):**
  - Feature số nhóm 1/3: as-of per transaction — không thống kê nào thấy t' ≥ t. (cần thống kê)
  - edge_attr GNN: lagged theo cửa sổ (val←train, test←train+val).
  - Cấu trúc graph: lũy tiến chuẩn Altman.
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
   khi xáo dữ liệu val (chỉ phụ thuộc train). chỉ nên làm graphsage thôi
5. **Nhóm 2** (`train_gnn.py`): screen → tune → chọn encoder.
6. Xuất embedding (+ mule optional). có thể bỏ chỉ làm graphsage+LSTM thôi
7. **Nhóm 3** (`train_hybrid.py`): XGBoost + ablation V0/V2.
8. Chọn pipeline tốt nhất; grouped SHAP.
9. (bonus) cross-test HI→LI; rồi đóng góp 2 (real-time).

---

## 8. Định vị khoa học & đóng góp

\*\*BaselineLi/ HI-Small (minority-F1, đã xác minh): phát triển dựa trên cách làm GraphSAGE+LSTM+XGBoost/lightGBM của amatriciana

**Đóng góp 1 (chính):** pipeline hybrid GraphSAGE-LSTM (cấp tài khoản) +
XGBoost/LightGBM (cấp giao dịch), kế thừa kiến trúc Amatriciana với 3 cải tiến
hướng real-time trên phần cứng phổ thông (VRAM 4GB):
(i) chuyển node-classification + split stratified ngẫu nhiên (có temporal leakage)
của Amatriciana thành transaction-classification + temporal split chặt;
(ii) thay multidimensional adjacency 384 step và feature centrality toàn cửa sổ
(closeness/eigenvector/clustering — O(V·E), không streaming) bằng
edge_attr gộp cạnh lagged + node_seq bucket ngày: mọi thống kê cập nhật
incremental O(1), tương thích streaming;
(iii) thay head MLP bằng gradient boosting.
Câu hỏi khoa học: đạt bao nhiêu % hiệu năng GNN edge-level nặng (Provably Powerful,
FraudGT) với chi phí thấp hơn (so chi phí: GFP, Quasi-temporal, Amatriciana).

**Đóng góp 2 (sau, systems):** đo và báo cáo chi phí suy luận theo từng giao dịch
của cả hai pipeline trên phần cứng phổ thông. KHÔNG làm kiến trúc 2 đường
nóng/nguội (chi phí hạ tầng lớn, không tạo đóng góp khoa học)

- Nhóm 1: streaming thật. Cập nhật as-of state O(1) → feature → LightGBM/xgboost
- Nhóm 3: hai tầng. Tầng online (tra embedding + as-of + LightGBM/xgboost) theo từng
  giao dịch; tầng offline (chạy lại encoder GraphSAGE) refresh theo cửa sổ.
- Kết luận nhắm tới: hybrid đạt X% hiệu năng với độ trễ online ngang nhóm 1,
  đổi lại cần refresh embedding mỗi N giờ. So chi phí: APAN/BRIGHT/blazingAML.

---

## 9. Trade-off & vấn đề mở

- **Limitation phải khai:** (a) edge_attr train-graph dùng chính cửa sổ train (không có
  dữ liệu trước) → mismatch ngữ nghĩa train/test — cùng tinh thần ExSTraQt batch;
  (b) cold-start: cạnh/account mới trong test có aggregate=0 — đúng vùng fraud hay nằm,
  cờ seen_before giúp model học điều này.
  — là phương pháp đề xuất; baseline GNN so cứng thì trích số công bố.
- so sánh mô hình baseline và đề xuất với các mô hình của bài báo khác
