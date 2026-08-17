# plan.md — Phát hiện giao dịch rửa tiền: so sánh 3 nhóm model ở cấp giao dịch

File context. Đọc đầu mỗi phiên để nắm trạng thái + kế hoạch.

folder extraqt-main là folder tải về từ bài báo exraqt có trong folder IBM AML paper/markdown

làm là phải tuân thủ cách chia, train trên quy chuẩn chung trong giới data science nếu không biết train/ split như thế nào
phải xem cách các bài báo làm không được chia bậy

pipeline: nhóm 1 (train_classical)+ nhóm 2(train-gnn-lstm-v3) -> kết quả train-hybrid

## 1. Tổng quan

**Đề tài (tạm):** Hệ thống phát hiện gian lận giao dịch tài chính thời gian thực (có thể sử dụng classifier hoặc graphsage+classfier)

**Hướng làm:** GNN mã hóa ở **cấp tài khoản** (nhẹ, ~515K node) nhưng **phân loại ở cấp giao dịch** (`Is Laundering`) để so trực tiếp với literature (Realistic synthetic money, Graph Feature Preprocessor, Provably powerful graph network, Amatriciana-những paper này đều có dạng markdown trong folder IBM AML paper/markdown).

**Khung thực nghiệm (đã chốt): 3 nhóm model, TẤT CẢ đánh giá ở cấp giao dịch, cùng split, cùng minority-F1.** Khác nhau ở mức độ dùng đồ thị:

- **Nhóm 1 — classical:** LR,DT, RF,XGBoost,lightGBM. Không message passing; phân loại từng giao dịch bằng transaction feature và as-of aggregate (tổng hợp feature không làm temporal leaky) nhóm 1 tune trong analysis_1.ipynb
- **Nhóm 2 — GNN:** GraphSAGE. Encoder trên graph tài khoản gộp cạnh (edge_attr lagged) + head phân loại từng giao dịch (end-to-end). bắt buộc làm graphsage những mô hình khác chỉ tham khảo
- **Nhóm 3 — hybrid:** edge/noedge-lstm + **classifier(XGBoost,LightGBM)** 4 seed nếu thành công đây là đóng góp 1

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

### Bước 7 — Nhóm 3 hybrid: GraphSAGE-LSTM(embedding) + XGBoost/LightGBM

**7a. Thực nghiệm hybrid** (`train_hybrid.py` ➕, chạy LOCAL)

Đầu vào có sẵn, không train lại encoder: `train_gnn.py` đã xuất embedding ngay
sau mỗi biến thể — 24 file `emb_{variant}_seed{k}_{split}.npy`, mỗi file
`[515_088, 32]` float32 (66 MB), đã tải về `scores/emb-lstm/`.
Ma trận nhóm 3 = `[90 cột txn_matrix | emb_src(32) | emb_dst(32) | 3 vô hướng]`,
ghép qua `txn_nodes.npy` theo đúng cách `train_gnn.py` làm (`nodes[split == s]`).

**Tập train bắt buộc là `train_b`, KHÔNG phải toàn bộ train.**
`WINDOWS["train"] = (["train_a"], [...])` — cả `edge_attr` lẫn `node_seq` của
train-graph chỉ tính từ `train_a` (`node_seq.py::verify` chứng minh bằng test xáo
dữ liệu). Với giao dịch `t ∈ train_a`, cửa sổ nguồn chứa chính `t` → embedding
hai đầu mút đã thấy `t'≥ t` → vi phạm §6. Chỉ `train_b` là causal.

- train_b: **1,523,920 dòng / 1,773 pos (0.116%)**
- toàn train: 3,046,861 dòng / 2,297 pos (0.0754%)
- val (1,015,602) và test (1,015,882) KHÔNG bị cắt — graph val dùng cửa sổ train,
  graph test dùng train+val. Hai nhóm đánh giá trên đúng cùng một tập.
- Ghi chú: train_b có fraud rate cao hơn toàn train 1.5×, đúng chiều distribution
  shift ở §3 → gần phân phối val/test hơn, bù một phần cho việc mất nửa dữ liệu.

**Lưới ablation — 5 nhánh × 2 booster × 4 seed = 40 lượt fit**, tất cả trên
`train_b`, cùng seed, cùng `PARAMS` đã tune ở nhóm 1 (không tune lại → khác biệt
duy nhất là feature):

| nhánh | cột | trả lời câu hỏi                                                               |
| ----- | --- | ----------------------------------------------------------------------------- |
| `V0`  | 90  | sàn — chính là nhóm 1 chạy trên train_b                                       |
| `V1e` | 93  | chỉ 3 vô hướng, bỏ 64 chiều thô — có cần embedding thô không?                 |
| `V2e` | 154 | embedding thô, encoder thắng (`sage_lstm`)                                    |
| `V3e` | 157 | thô + 3 vô hướng — boosting khai thác hết embedding thô chưa?                 |
| `V2n` | 154 | embedding thô, `sage_noedge_lstm` — edge_attr còn giá trị dưới head boosting? |

Ba cột vô hướng = `cosine(u,v)`, `‖u−v‖₂`, `dot(u,v)`. Lý do: cây chia trục-song-song,
không biểu diễn được tích vô hướng của hai vector 32 chiều. Tiền lệ: binary operator
của node2vec (Grover & Leskovec 2016).

Quy ước bắt buộc:

- Ghép seed theo cặp: booster seed `k` dùng `emb_..._seed{k}_*`. **KHÔNG trung bình
  embedding qua seed** — mỗi lần train sinh một không gian ẩn khác (hoán vị/xoay),
  cộng trung bình sẽ triệt tiêu tín hiệu. (Trung bình _điểm dự đoán_ thì hợp lệ.)
- `V0` phải chạy lại trên train_b, không tái dùng số của `results.csv`.
- Nhưng KHÔNG được bỏ bản nhóm 1 trên 100% train — nhóm 1 không bị ràng buộc lagged
  nên train đủ là quyền chính đáng của nó. Cắt baseline để đóng góp trông mạnh là
  lỗi phản biện bắt ngay. Báo cáo cả hai, tách hai khối bảng.

Đầu ra: `results-hybrid.csv` (xoá trước mỗi lần chạy — `log_result` ghi mode="a"),
`scores/{arm}_{booster}_seed{k}_{split}.npy`, `models/{arm}_{booster}_seed{k}.json|joblib`.
Hiệu ghép cặp theo seed phải in ra: `V2e−V0`, `V1e−V0`, `V2e−V1e`, `V3e−V2e`, `V2e−V2n`.
Với 4 seed và std cỡ 0.01 (mức của nhóm 2), hiệu **dưới ~0.01 phải phát biểu là
"không phân biệt được"**, không phải "bằng nhau", càng không phải "thua".

Chi phí đo được: RAM đỉnh ~4.7 GB, ~20 phút trên máy local.

**7b. Web demo** (dựng song song trong lúc chạy 7a, bằng mô hình nhóm 1)

Khung web gồm: tùy chọn khoảng ngày, tùy chọn mô hình, biểu đồ thống kê giao dịch,
thống kê feature của 1 node trong khoảng thời gian, bảng tra cứu embedding, lọc giao
dịch gian lận/không gian lận, nút chi tiết hiện pattern và SHAP.

Quy trình đo latency — tách hai chế độ, KHÔNG gộp:

- **Chế độ duyệt:** đọc score/feature đã precompute → UI mượt khi kéo slider.
  Đây là phần demo cho hội đồng xem.
- **Chế độ đo** (`benchmark_latency.py`): replay tuần tự theo Timestamp, tính as-of
  state LIVE, chấm điểm từng giao dịch. Chạy một lần, xuất bảng số.

Ràng buộc khi đo:

- batch size = 1 (predict cả tập rồi chia = đo throughput theo lô, không phải latency)
- dùng numpy array, KHÔNG dựng DataFrame 1 dòng mỗi giao dịch
- warm-up: bỏ ~5.000 giao dịch đầu trước khi ghi số
- tách thành phần: cập nhật as-of / dựng vector / tra embedding (nhóm 3) / predict
- báo cáo p50/p95/p99 + throughput, đơn luồng
- đối chiếu ngưỡng nghiệp vụ CÓ TRÍCH DẪN (vd. độ trễ luồng authorization thẻ)
- **toàn bộ đo trên máy A (local)**, xem §8 Môi trường thực nghiệm

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

**Môi trường thực nghiệm (phải khai trong luận văn):**

- **Máy A — local:** i5-13420H (8 nhân/12 luồng), 16 GB RAM, RTX 2050 4 GB.
  Chạy nhóm 1, nhóm 3, và TOÀN BỘ phần đo độ trễ/throughput.
- **Máy B — Kaggle:** 1× Tesla T4 (trong 2 card được cấp; `train_gnn.py` chỉ dùng
  `cuda:0`), 4 vCPU, ~29 GB RAM. Chỉ huấn luyện encoder nhóm 2.
  Lý do có số: `VRAM đỉnh 3.52 GB` vượt bộ nhớ khả dụng của card 4 GB.
- Số **chất lượng** (f1_minority, PR-AUC, recall@FPR, precision@k) không phụ thuộc
  phần cứng — cùng split, cùng `txn_matrix`, cùng `metrics.py`, cùng quy trình dò
  ngưỡng trên val → so chéo ba nhóm hợp lệ.
- Số **chi phí** (`train_time_s`) phụ thuộc phần cứng → ghi kèm `"device"` trong
  chuỗi `params`, KHÔNG đặt cạnh nhau để so chéo nhóm. Mọi luận điểm chi phí đo
  lại trên máy A.

**Đóng góp 1 (chính):** pipeline hybrid GraphSAGE-LSTM (cấp tài khoản) +
XGBoost/LightGBM (cấp giao dịch), kế thừa kiến trúc Amatriciana với 3 cải tiến:
(i) chuyển node-classification + split stratified ngẫu nhiên (có temporal leakage)
của Amatriciana thành transaction-classification + temporal split chặt;
(ii) thay multidimensional adjacency 384 step và feature centrality toàn cửa sổ
(closeness/eigenvector/clustering — O(V·E), không streaming) bằng edge_attr gộp
cạnh lagged + `node_seq` bucket 6 giờ (K=8, tức 2 ngày cuối cửa sổ nguồn):
mọi thống kê cập nhật incremental O(1), tương thích streaming;
(iii) thay head MLP bằng gradient boosting.

**Phát biểu chính xác về phần cứng** (không được viết "train được trên VRAM 4 GB"):
huấn luyện encoder là bước **offline, định kỳ**, cần GPU 16 GB (đo được: đỉnh
3.52 GB, quá tầm RTX 2050); còn **suy luận từng giao dịch** — tra embedding đã lưu

- cập nhật as-of + gradient boosting — không đụng GPU và chạy trên phần cứng phổ
  thông. Đóng góp hướng real-time nằm ở tầng suy luận, không ở tầng huấn luyện.

**Cách chứng minh đóng góp:** lưới ablation 5 nhánh ở Bước 7a. Kết luận rút ra từ
hiệu ghép cặp theo seed, không từ so số tuyệt đối với nhóm 1 (khác lượng dữ liệu train).

- `V2e − V0` > 0: embedding cấp tài khoản làm classifier cấp giao dịch tốt lên.
- `V2e − V1e` > 0: 64 chiều thô mang thông tin không quy về khoảng cách giữa hai node
  — nếu ≈ 0 thì encoder LSTM chưa biện minh được, phải xem lại đóng góp.
- `V2e − V2n`: edge_attr có còn giá trị khi head là boosting thay vì MLP.
  (Nhóm 2 với head MLP: test f1 0.5205 so 0.4881 — chênh 0.032, std 0.0133.)
- `V3e − V2e`: giới hạn biểu diễn trục-song-song của cây có thật hay không.

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
