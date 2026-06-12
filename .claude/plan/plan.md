# plan.md — Hệ thống phát hiện giao dịch gian lận đa tầng (GNN + XGBoost)

File context cho dự án. Đọc file này đầu mỗi phiên để nắm trạng thái và kế hoạch.

---

## 1. Tổng quan dự án

**Tên đề tài:** Hệ thống phát hiện giao dịch gian lận đa tầng thời gian thực, kết hợp GNN và XGBoost (AML — Anti-Money Laundering).

**Hướng tiếp cận (hướng A đã chốt):** phân loại ở cấp **tài khoản (node classification)**, không phải cấp giao dịch, để giảm quy mô đồ thị từ ~5 triệu node giao dịch xuống ~515K node tài khoản — khả thi với phần cứng hạn chế.

**Kiến trúc 2 tầng:**

- **Tầng 1 — GNN (GraphSAGE):** học embedding cấu trúc mạng cho mỗi tài khoản, dự đoán điểm "mule" của tài khoản.
- **Tầng 2 — XGBoost (CPU):** phân loại `Is Laundering` ở cấp giao dịch, dùng feature giao dịch + embedding + điểm mule của hai đầu giao dịch từ Tầng 1.

**Dataset:** IBM AMLworld (Altman et al. 2023).

- Train/eval chính trên **HI-Small** (`dataset_high/`) — tỉ lệ illicit cao hơn, nhiều pattern, dễ huấn luyện/đánh giá GNN.
- **LI-Small** (`dataset_small/`) để **cross-dataset test** (train HI → test LI), mô phỏng việc áp model đã kiểm nghiệm lên dữ liệu mới.

**Ràng buộc phần cứng:** RAM 16GB, GPU RTX 2050 VRAM 4GB. Mọi quyết định kiến trúc (gộp cạnh, neighbor sampling, mixed precision) đều xoay quanh ràng buộc này.

---

## 2. Cấu trúc thư mục & file hiện có

```
D:\ct551_v2\
├─ dataset_high/                 # HI-Small (dataset chính)
│  ├─ HI-Small_Trans.csv         # ~5.08M giao dịch (cạnh). Nhãn gốc 'Is Laundering' ở đây
│  ├─ HI-Small_accounts.csv      # thông tin tài khoản (Bank Name, Bank ID, Account Number, Entity)
│  └─ HI-Small_Patterns.txt      # 8 dạng pattern rửa tiền — DÀNH cho phân tích, KHÔNG dùng gán nhãn
├─ dataset_small/                # LI-Small (dùng cross-test)
│  ├─ LI-Small_Trans.csv
│  ├─ LI-Small_accounts.csv
│  └─ LI-Small_Patterns.txt
├─ AccountFraudLabel.csv         # nhãn node: (Account, Bank ID, IsLaundering) — 515,088 dòng
├─ nodelabel.py                  # script tạo AccountFraudLabel.csv từ Trans + accounts
├─ temporal_split.py             # script chia train/val/test theo thời gian 60/20/20
├─ split_by_day.csv              # bảng tra cứu gọn: ngày → split + số liệu (18 dòng)
├─ check_duplicate.ipynb         # kiểm tra account trùng mã giữa các bank
├─ README.docx                   # ghi chú quá trình (nháp)
└─ CLAUDE.md                     # file này
```

**Cột của `HI-Small_Trans.csv`:** `Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`. (Hai cột tên "Account" → pandas tự đổi thành `Account` và `Account.1`.)

---

## 3. Số liệu dataset đã xác minh (HI-Small)

> Đây là số đếm thật trên data, không phải ước lượng.

- Tổng giao dịch: **5,078,345**.
- Khoảng thời gian: 2022/09/01 → 2022/09/18, nhưng **~99% giao dịch nằm trong ngày 1–10**; ngày 11–18 gần như rỗng.
- Giao dịch laundering: **5,177 / 5.08M ≈ 0.10%** (mất cân bằng cực mạnh ở cấp cạnh).
- Tài khoản (node) trong `AccountFraudLabel.csv`: **515,088**, trong đó **6,357 dương ≈ 1.23%** (mất cân bằng ở cấp node, đỡ hơn cấp cạnh).
- `Patterns.txt` có đủ 8 loại: fan-in, fan-out, bipartite, cycle, gather-scatter, scatter-scatter, stack, random.

**Lưu ý leading zero:** `From Bank` / `To Bank` trong Trans.csv có số 0 ở đầu (vd `001`, `0010`), còn `Bank ID` trong accounts.csv là số. Khi join phải đọc **dạng string** để không ghép sai node.

---

## 4. Quy trình split đã chốt (đã triển khai trong `temporal_split.py`)

**Nguyên tắc:** chia theo **thời gian (chronological)**, KHÔNG random — để tránh leakage tương lai và phản ánh đúng lúc deploy (train quá khứ, test tương lai). Cắt theo **ranh giới đầu ngày (00:00)** vì hệ thống refresh node embedding theo ngày; cắt giữa ngày sẽ gây leakage.

**Kết quả split 60/20/20 (snap theo ngày → tỉ lệ thực 54.5/19/26.5):**

| Split | Ngày        | Số giao dịch | %      | Fraud | Fraud rate |
| ----- | ----------- | ------------ | ------ | ----- | ---------- |
| train | 09/01–09/05 | 2,766,832    | 54.48% | 1,999 | 0.0722%    |
| val   | 09/06–09/07 | 964,840      | 19.00% | 1,028 | 0.1065%    |
| test  | 09/08–09/18 | 1,346,673    | 26.52% | 2,150 | 0.1597%    |

- Ranh giới: train|val = đầu ngày **2022/09/06**; val|test = đầu ngày **2022/09/08**.
- Đã assert không chồng lấn thời gian giữa các tập.
- **Distribution shift:** fraud rate tăng dần train→test (~2.2×). Đây là lý do bài toán cần continual learning, và là lý do dùng PR-AUC/Recall thay vì Accuracy.

> Khớp **nguyên tắc** với split của paper gốc IBM (Altman et al. 2023): 60-20-20 temporal theo 2 mốc t1, t2; dựng graph lũy tiến (train-graph chỉ chứa train edges; val-graph = train+val; test-graph = all; chỉ eval trên index tương ứng). **Khác biệt nhỏ:** paper cắt theo tỉ lệ index giao dịch chính xác (t1, t2 có thể rơi giữa ngày), còn ta snap theo đầu ngày → 54.5/19/26.5. Chấp nhận được vì ta tự chạy lại mọi baseline trên cùng split; có thể thêm bản cắt theo index làm ablation.

---

## 5. Tiến trình hiện tại

**Đã xong:**

- [x] `nodelabel.py` → `AccountFraudLabel.csv` (gán nhãn node từ nhãn cạnh).
- [x] `temporal_split.py` → logic split temporal 60/20/20 (đã sửa: tên file, giữ string bank/account, dùng `.iloc`, in thống kê, assert no-leakage).
- [x] `split_by_day.csv` → bảng tra cứu ngày → split.
- [x] Khảo sát cách split của các paper (IBM dataset paper, review 2503.24259, BeGin, collaborative AML) và chốt 4 protocol thực nghiệm (xem mục 7).
- [x] `temporal_split_index.py` → biến thể thứ hai của protocol 1: cắt-theo-index 60/20/20 (chuẩn Altman). Đã qua review (PASS sau 1 vòng sửa). Tie-breaking đặt ranh giới theo GIÁ TRỊ Timestamp (dồn trọn nhóm cùng thời điểm về một phía) để chặn leakage hai chiều; có assert no-overlap strict + assert leading-zero kiểm nội dung. Output `dataset_high/HI-Small_Trans_split_index.csv` (60.00/20.00/20.00). Mục tiêu thực nghiệm đã chốt theo "cách 3" (mục 7).

**Đang/kế tiếp:** xây Tầng 1 (GNN) — pipeline cố định mục 8 bước 4–8, chạy được trên CẢ HAI biến thể split (snapshot-ngày `HI-Small_Trans_split.csv` và cắt-index `HI-Small_Trans_split_index.csv`) để so sánh protocol.

---

## 6. Thiết kế kỹ thuật Tầng 1 — GNN (đã chốt)

### Bước 0 — Nhãn cho node (nền tảng)

Nhãn `Is Laundering` ở trên cạnh; phải suy ra nhãn node. **Quy tắc:** tài khoản là **dương (mule)** nếu xuất hiện (gửi HOẶC nhận) trong ≥1 giao dịch có `Is Laundering = 1`. Lấy thẳng từ `Trans.csv`, không dùng `Patterns.txt`. (Thực tế đo được: 6,357/515,088 ≈ 1.23% dương.) **Định danh node bằng khóa tuple `(Bank, Account)`** (Bước 1) — `nodelabel.py` cần sửa lại theo tuple (hiện gán theo `Account` đơn).

### Bước 1 — Danh tính node (ĐÃ CHỐT: dùng khóa tuple)

Cùng mã `Account` có thể thuộc bank khác nhau → **dùng khóa tuple** `(Bank, Account)`: bên gửi `(From Bank, Account)`, bên nhận `(To Bank, Account.1)`. Đã xác nhận qua `check_duplicate.ipynb`: có account trùng mã giữa các bank → khóa tuple là bắt buộc, không phải tùy chọn. Mọi bước sau (nhãn node, feature, graph, embedding) đều định danh node bằng tuple này. Đọc bank/account dạng **string** để giữ leading zero.

### Bước 2 — Chia tập theo thời gian (làm TRƯỚC feature)

Đã làm (mục 4). Hệ quả: mọi feature/embedding chỉ tính từ dữ liệu trong cửa sổ train; tài khoản ở test dùng lại embedding học từ train (inductive). Khớp đúng cơ chế "refresh embedding theo batch" của kiến trúc.

### Bước 3 — Feature engineering cho node (chỉ tính trên train)

~20–30 feature số/node, nhóm theo nghiệp vụ:

- **Dòng tiền:** tổng gửi, tổng nhận, số GD gửi/nhận, mean/std số tiền; quan trọng nhất `net_flow = nhận − gửi` (mule ≈ 0 vì tiền chỉ chảy qua).
- **Đa dạng đối tác:** out-degree riêng biệt, in-degree riêng biệt → bắt fan-out / fan-in.
- **Xuyên ngân hàng:** số bank tương tác, tỉ lệ GD xuyên bank.
- **Tiền tệ:** số loại currency, tỉ lệ GD có `Receiving ≠ Payment Currency`.
- **Hình thức thanh toán:** tỉ lệ mỗi `Payment Format` (Wire/Cheque/ACH/Credit Card...).
- **Thời gian (velocity):** số ngày hoạt động, GD/ngày, khoảng cách thời gian trung bình; nâng cao: thời gian trung bình nhận→đẩy đi.
- **Số tròn:** tỉ lệ GD số tiền tròn hoặc ngay dưới ngưỡng báo cáo (structuring).

**Bắt buộc:** áp `log1p` cho feature tiền/đếm (lệch phải nặng) trước khi scale; fit scaler (z-score/robust) **chỉ trên train** rồi transform test (fit toàn bộ = leakage).

### Bước 4 — Dựng graph

- **Node:** mỗi tài khoản (khóa tuple Bước 1) + vector feature Bước 3.
- **Cạnh:** **gộp** nhiều giao dịch giữa cùng cặp (gửi→nhận) thành 1 cạnh; edge feature = tổng tiền + số lần. Nhẹ RAM; chấm điểm từng GD đẩy sang XGBoost.
- **Hướng cạnh:** v1 làm đối xứng (thêm cạnh ngược). v2 (sau) tách aggregation chiều vào/ra.
- Toàn graph (~515K node, vài triệu cạnh, ~25 chiều feature) giữ trên **CPU/RAM**; chỉ subgraph mỗi batch lên GPU.

### Bước 5 — Kiến trúc GNN

- **GraphSAGE 2 layer** (inductive + chạy tốt với neighbor sampling). 2 hop đủ bắt fan-in/fan-out/chuỗi ngắn. Không sâu hơn (tốn VRAM + oversmoothing).
- Khung:

```python
class SageNet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)   # output = embedding
        self.head  = nn.Linear(hidden, 1)        # phân loại mule
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        emb = self.conv2(h, edge_index)          # 64-chiều cho XGBoost
        return self.head(emb), emb
```

- Cấu hình RTX 2050: `hidden=64` (tăng 128 nếu dư VRAM); `NeighborLoader` `num_neighbors=[15,10]`, `batch_size=512`; mixed precision (`torch.cuda.amp` + GradScaler); Adam lr 0.01, dropout 0.3, weight decay nhẹ.

### Bước 6 — Mất cân bằng (phần khó nhất)

Với ~1.23% node dương, train ngây thơ → đoán "tất cả sạch", accuracy 98%+ vô dụng.

1. **Phạt nặng bỏ sót:** `BCEWithLogitsLoss(pos_weight≈ âm/dương)` (cỡ vài chục–trăm); hoặc focal loss.
2. **Metric đúng:** bỏ accuracy. Chính: **AUC-PR (average precision)**; kèm precision/recall và **precision@k**. Báo cáo theo ngôn ngữ nghiệp vụ "recall ở mức false-positive chấp nhận được".

### Bước 7 — Bàn giao sang XGBoost

Xuất embedding 64 chiều (output `conv2`, trước head) cho mọi tài khoản. Vector vào XGBoost cho mỗi giao dịch:
`[feature giao dịch] + [emb tài khoản gửi] + [emb tài khoản nhận] + [điểm mule gửi/nhận]`.
XGBoost (CPU) học `Is Laundering` cấp giao dịch (nhãn gốc).
Metric đối chiếu benchmark IBM: **minority-class F1 cấp giao dịch** (kèm PR-AUC). Số tham chiếu HI-Small (temporal split, edge-level, từ paper — chỉ để định vị, không so cứng): GIN 28.7, PNA 56.8, GFP+XGBoost 63.2, Multi-PNA+EU 68.2.
**Chống leakage:** embedding cho giao dịch test phải tính từ **graph train**, không tính lại bằng dữ liệu test.

---

## 7. Bốn protocol split để so sánh thực nghiệm (theo yêu cầu giảng viên)

Mục tiêu (cập nhật 2026-06-11 theo yêu cầu giảng viên — "chọn phương pháp tốt nhất", diễn giải theo cách 3): **trong nhóm các protocol KHÔNG leakage, chọn protocol cho hiệu năng + độ ổn định cao nhất.** Cố định mọi thứ khác (backbone, feature, hyperparameter), chỉ đổi protocol.

- **Nhóm ứng viên được chọn** (đều hợp lệ, không leakage thời gian): protocol 1 (temporal hold-out, gồm 2 biến thể snapshot-theo-ngày vs cắt-theo-index), protocol 3 (domain-incremental), protocol 4 (class-incremental). So các protocol này với nhau bằng PR-AUC / Recall / F1-minority (+ AF/AP cho continual) để chọn ra cái "tốt nhất".
- **Protocol 2 (random/stratified) KHÔNG nằm trong nhóm ứng viên** — nó leakage thời gian/pattern nên không bao giờ được chọn. Vai trò duy nhất: chứng cứ phản chứng để định lượng mức thổi phồng (random cho điểm cao hơn bao nhiêu so với temporal, và đó là ảo do leakage). Đây là lý do plan vẫn giữ protocol 2 dù không chọn nó.
- Lưu ý phương pháp luận: "tốt nhất" KHÔNG phải "metric tuyệt đối cao nhất" (nếu thế sẽ luôn là random/leakage). "Tốt nhất" = cao nhất TRONG SỐ các protocol đã loại leakage.

1. **Temporal hold-out 60/20/20** (chuẩn Altman) — baseline tĩnh. ✅ đã có.
2. **Random / stratified split** — baseline "ngây thơ" để chứng minh nó thổi phồng kết quả vs temporal. Dẫn chứng literature: Amatriciana (IEEE ICDMW 2024, arXiv 2506.00654) chia **stratified theo node trên cùng khung thời gian** (train/test cùng giai đoạn), không bàn temporal leakage → tái lập đúng cách chia này để định lượng mức thổi phồng.
3. **Domain-incremental theo thời gian** (task = 1 ngày, ngày 1–10) — continual; khớp thiết kế streaming + cách review làm cho Elliptic.
4. **Class-incremental theo pattern** (dùng `Patterns.txt`, thêm 1 pattern/task) — continual; tái lập cách review (2503.24259) làm cho IBM, để so head-to-head. Lưu ý: review loại bỏ 1,968/5,177 giao dịch laundering không gắn pattern ("not classified") — phải quyết định cách xử lý tương đương và ghi rõ.

Metric: static → PR-AUC / Recall / F1-minority; continual → thêm AF (average forgetting), AP (average performance). Không so cứng số với paper — chỉ tự chạy lại các protocol trên đúng setup rồi so với nhau.

---

## 8. Thứ tự công việc (lộ trình)

1. Load + tối ưu dtype (giữ string bank/account).
2. Kiểm tra danh tính node (tuple account và bank ID).
3. Chia tập theo thời gian. ✅
4. Feature engineering node (pandas, chỉ trên train) — hiệu quả trên 5M dòng.
5. Dựng graph PyG (cạnh gộp, đối xứng).
6. Train GraphSAGE (NeighborLoader + AMP + pos_weight).
7. Xuất embedding.
8. Lắp dataset XGBoost, train, đánh giá AUC-PR.
9. (Sau) đấu nối replay Kafka/Redpanda theo `Timestamp` để mô phỏng real-time.

---

## 9. Vấn đề mở / cần lưu ý

- **Label leakage (nhẹ, chấp nhận v1):** `AccountFraudLabel.csv` gán fraud theo toàn timeline. V1 giữ nhãn global; v2 có thể tính nhãn as-of-time làm ablation.
- **Leakage feature/embedding (đã chặn):** chỉ tính từ cửa sổ train; fit scaler trên train.
- **Patterns.txt:** chỉ dùng cho protocol 4 và phân tích kết quả, không dùng gán nhãn node.
- **Tie-breaking khi tái dùng split cho dataset khác (technical debt):** `temporal_split_index.py` đặt ranh giới theo giá trị Timestamp; logic này cần kiểm lại edge case khi không có dòng nào vượt boundary (vd áp lên LI-Small cross-test) trước khi tái dùng. Không ảnh hưởng HI-Small.

---

## 10. Định vị khoa học & hướng đóng góp (cập nhật 2026-06-11, đã xác minh từ nguồn)

### Bản đồ literature trên IBM AML

- **Altman et al. 2023** (NeurIPS D&B, arXiv 2306.16424): tạo dataset bằng generator **AMLworld** (kế thừa AMLSim, calibrate theo Federal Reserve — KHÔNG liên quan AliPay). Chuẩn: temporal 60/20/20 + progressive graph; task chuẩn = edge classification; metric = minority-class F1.
- **Egressy et al. 2024** (AAAI, arXiv 2306.11586): Multi-GNN — reverse message passing + port numbering + ego IDs; SOTA HI-Small F1 68.16. Gain chủ yếu từ reverse MP + ports (gợi ý cho v2: tách aggregation chiều vào/ra).
- **Blanuša et al. 2024** (ICAIF, arXiv 2402.08593): GFP — trích đặc trưng subgraph (fan-in/out, cycle, scatter-gather) dạng streaming trên CPU + XGBoost; F1 63–65; 12–23k giao dịch/s, latency ~30ms. Có sẵn trong thư viện **Snap ML** (pip) → dùng được làm baseline.
- **Deprez et al. 2025** (arXiv 2503.24259): review + benchmark continual graph learning cho AML; class-incremental theo pattern trên HI-Small, graph tĩnh; backbone chỉ GCN; forgetting nặng (AP ≈ 0.21). Nguồn cho protocol 4.
- **Amatriciana 2024** (IEEE ICDMW, arXiv 2506.00654): node classification + LSTM temporal encoder trên HI-Small; chia **stratified theo node, KHÔNG temporal** → kết quả không so sánh được với nhánh chuẩn. Nguồn cho protocol 2.
- **LAS-GNN** (ACM ICAIF '25): LSTM aggregator cho motif temporal, node-level; thí nghiệm trên dữ liệu synthetic tự sinh (chưa xác nhận chạy HI-Small; full text paywalled).
- **RIPPLE++** (arXiv 2601.12347): framework hạ tầng GNN inference incremental trên đồ thị tiến hóa (systems paper, ICDCS 2025 mở rộng); KHÔNG chạy dữ liệu tài chính — fraud chỉ là motivation.
- **Collaborative AML** (WWW '25, arXiv 2502.19952): mining scatter-gather xuyên tổ chức có bảo mật; bộ Alipay-ECB là **dữ liệu thật**, không phải synthetic; dùng AMLworld HI/LI làm benchmark phụ. (Đính chính: không tồn tại "SIMAML".)

### Khoảng trống → 3 đóng góp của đề tài

1. **So sánh protocol chia dữ liệu (đóng góp chính):** literature chia rất không nhất quán (temporal chuẩn vs stratified-node vs pattern-incremental). Chưa có nghiên cứu nào cố định backbone + feature và CHỈ đổi protocol trên IBM AML để định lượng mức thổi phồng kết quả và mức quên. Mục 7 chính là nghiên cứu đó.
2. **Pipeline 2 tầng node-embedding → XGBoost cấp giao dịch trên temporal split chuẩn:** NVIDIA blueprint mô tả pattern này nhưng không công bố số trên benchmark công khai; bài IEEE OTCON 2025 cùng ý tưởng nhưng hội nghị nhỏ, số liệu không kiểm chứng được (paywalled, không nêu dataset/split trong abstract). Chưa ai công bố đánh giá nghiêm túc pipeline này trên IBM AML với split chuẩn, so head-to-head với GFP+XGBoost và GNN end-to-end. Câu hỏi khoa học: embedding tài khoản từ GNN nhẹ (~515K node) + XGBoost đạt được bao nhiêu % hiệu năng của GNN edge-level (~5M edge) với chi phí thấp hơn bao nhiêu?
3. **Real-time / incremental (v2):** RIPPLE++ chứng minh inference incremental khả thi nhưng chưa ai áp lên AML. Hướng mở: replay theo `Timestamp` + cập nhật embedding incremental thay vì refresh theo ngày.

### Baseline bắt buộc cho đóng góp 2

| Baseline                               | Vai trò                                           | Chi phí |
| -------------------------------------- | ------------------------------------------------- | ------- |
| XGBoost thuần (feature giao dịch)      | sàn dưới (GFP paper: ~19.75 F1)                   | CPU     |
| GFP (Snap ML) + XGBoost                | đối thủ chính cùng triết lý "graph feature + GBM" | CPU     |
| GraphSAGE embedding + XGBoost (đề tài) | pipeline đề xuất                                  | GPU 4GB |
| GIN/PNA/Multi-PNA edge-level           | chỉ trích dẫn số công bố, không tự chạy           | —       |

### Việc rút ra cho bước hiện tại (chia dữ liệu)

- Giữ split temporal đã làm — đúng chuẩn nhánh benchmark IBM.
- Viết thêm script chia stratified-node kiểu Amatriciana (protocol 2), dùng chung format đầu ra với `temporal_split.py` để pipeline phía sau không đổi.
- Mọi kết quả xuất ra phải ghi rõ protocol — đây là biến độc lập của nghiên cứu.

---
