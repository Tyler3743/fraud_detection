# plan.md — Hệ thống phát hiện giao dịch gian lận đa tầng (GNN + XGBoost)

File context cho dự án. Đọc file này đầu mỗi phiên để nắm trạng thái và kế hoạch.

---

## 1. Tổng quan dự án

**Tên đề tài:** Hệ thống phát hiện gian lận giao dịch tài chính thời gian thực kết hợp GNN và XGBoost

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
- [x] Khảo sát cách split của các paper (IBM dataset paper, review 2503.24259, BeGin, collaborative AML) và chốt cách chia dữ liệu (xem mục 7).
- [x] `temporal_split_index.py` → cắt-theo-index 60/20/20 (chuẩn Altman) — **split chính**. Đã qua review (PASS sau 1 vòng sửa). Tie-breaking đặt ranh giới theo GIÁ TRỊ Timestamp (dồn trọn nhóm cùng thời điểm về một phía) để chặn leakage hai chiều; có assert no-overlap strict + assert leading-zero kiểm nội dung. Output `dataset_high/HI-Small_Trans_split_index.csv` (60.00/20.00/20.00).

**Đang/kế tiếp:** xây Tầng 1 (GNN) — pipeline mục 8 bước 4–8, báo cáo kết quả chính trên split-index. (Split-day giữ làm baseline cho đóng góp 2 real-time, không phải để so sánh protocol.)

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

## 7. Cách chia dữ liệu cho thực nghiệm (đã chốt — cập nhật 2026-06-15)

**Quyết định khung:** đã BỎ ý tưởng "đo độ nhạy / leakage audit giữa các protocol" khỏi phạm vi đóng góp. Lý do: giữa hai biến thể temporal (cắt-index vs snap-ngày) không có cái nào "tốt hơn" — cả hai đều đúng, chênh lệch chỉ phản ánh tỷ lệ chia và thành phần tập test khác nhau, giá trị khoa học mỏng. Không có báo cáo nào chứng minh snap-ngày cho model tốt hơn; cái literature chứng minh là "temporal tốt hơn random" về độ trung thực (random thổi phồng metric ~5–15% do lộ thông tin tương lai).

**Split chính (báo cáo mọi kết quả chính ở đây): cắt theo index 60/20/20 — chuẩn Altman.** Mỏ neo bắt buộc để số so được với Altman 2023 / Egressy 2024 / GFP 2024. ✅ đã có (`HI-Small_Trans_split_index.csv`).

**Thí nghiệm phụ (KHÔNG phải đóng góp): so với random split.** Chạy thêm 1 lần chia ngẫu nhiên để minh họa nó thổi phồng metric so với temporal split. Mục đích: chứng minh phương pháp đánh giá nghiêm túc, có literature hậu thuẫn. Rẻ (1 lần chạy, không đổi pipeline). Đây là phần duy nhất còn giữ lại từ ý tưởng "đo độ nhạy" cũ, vì nó có giá trị thật (temporal vs random), khác với so 2 biến thể temporal.

**Split-day (snap đầu ngày): hạ xuống baseline cho nhánh real-time (đóng góp 2).** Refresh embedding theo ngày là điểm xuất phát để so với cập nhật incremental. KHÔNG dùng báo cáo metric chính, KHÔNG bán như "split tốt hơn". ✅ đã có (`HI-Small_Trans_split.csv`).

**Đã hoãn (defer, không nằm trong scope):**

- **Stratified-node split (so Amatriciana):** Amatriciana dùng TOÀN graph, không cắt thời gian → leaky so với deploy. Để sau.
- **Class-incremental theo pattern (so review KU Leuven):** cần `Patterns.txt` gán nhãn lại → đổi Bước 0 + xử lý node thuộc nhiều pattern. Scope nặng, để v2.

Metric chính: PR-AUC / Recall / F1-minority trên split-index. Ghi rõ split trong mọi kết quả.

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
   8b. Ablation 3 biến thể (V0 feature giao dịch / V1 +điểm mule / V2 +embedding) → bằng chứng đóng góp 1; kèm 1 hình grouped SHAP minh họa (xem mục 10, đóng góp 1).
9. (Sau) đấu nối replay Kafka/Redpanda theo `Timestamp` để mô phỏng real-time.

---

## 9. Vấn đề mở / cần lưu ý

- **Label leakage (nhẹ, chấp nhận v1):** `AccountFraudLabel.csv` gán fraud theo toàn timeline. V1 giữ nhãn global; v2 có thể tính nhãn as-of-time làm ablation.
- **Tie-breaking khi tái dùng split cho dataset khác (technical debt):** `temporal_split_index.py` đặt ranh giới theo giá trị Timestamp; logic này cần kiểm lại edge case khi không có dòng nào vượt boundary (vd áp lên LI-Small cross-test) trước khi tái dùng. Không ảnh hưởng HI-Small.

---

## 10. Định vị khoa học & hướng đóng góp (cập nhật 2026-06-11, đã xác minh từ nguồn)

### Literature liên quan (tóm tắt ngắn)

**Mốc so sánh trực tiếp:**

- **Altman 2023** (tạo ra dataset IBM AML): đặt "luật chơi" chuẩn — chia dữ liệu theo thời gian 60/20/20, phân loại ở cấp giao dịch, đo bằng F1 lớp thiểu số. Mình bám chuẩn này để số so được với họ.
- **Egressy 2024**: model GNN mạnh nhất hiện nay trên HI-Small (F1 68.16). Dùng làm mốc trên để biết pipeline mình đạt bao nhiêu % so với đỉnh.
- **Blanuša 2024 (GFP)**: trích đặc trưng đồ thị (đếm fan-in/out, chu trình...) rồi đưa vào XGBoost; chạy nhanh trên CPU, có sẵn trong thư viện Snap ML (cài bằng pip). Đây là đối thủ trực tiếp cùng triết lý "đặc trưng đồ thị + cây quyết định", và mình chạy lại được để so head-to-head.
- **Amatriciana 2025**: rất giống hướng A của mình (GraphSAGE cấp tài khoản). Khác biệt then chốt: họ dùng toàn bộ đồ thị, không cắt theo thời gian → lộ thông tin tương lai. Vừa là tham chiếu gần nhất, vừa là ví dụ minh chứng vì sao phải chia theo thời gian.

**Chỉ cần cho hướng v2 (real-time / học liên tục):** RIPPLE++ 2025 (kỹ thuật chạy GNN incremental trên đồ thị thay đổi liên tục, chưa ai áp lên dữ liệu tài chính); review continual KU Leuven 2025 (tổng hợp các kỹ thuật học liên tục); GFP/NVIDIA (cho con số throughput/latency tham khảo).

### 2 đóng góp của đề tài (cập nhật khung 2026-06-15)

1. **Pipeline 2 tầng node-embedding → XGBoost cấp giao dịch (ĐÓNG GÓP CHÍNH):** câu hỏi khoa học = embedding tài khoản từ GNN nhẹ (~515K node) + XGBoost đạt bao nhiêu % hiệu năng của GNN edge-level (~5M edge) với chi phí thấp hơn bao nhiêu? Báo cáo trên split-index chuẩn Altman. **Đối thủ thật (kiểm chứng được):** GFP+XGBoost (Snap ML, chạy lại được) và GNN end-to-end Egressy (số đã công bố, F1 68.16). **Chỉ là citation, KHÔNG dùng làm baseline:** NVIDIA blueprint (không công bố số benchmark công khai) và IEEE OTCON 2025 (paywalled, số không kiểm chứng được) — neo đóng góp vào cái chạy lại được.
   - **Bằng chứng "embedding GNN có đóng góp" (chốt 2026-06-16):** xương sống là **ablation** (đo giá trị biên), SHAP chỉ là minh họa phụ.
     - **Ablation 3 biến thể** trên cùng split-index, cùng nhãn, chỉ đổi vector vào XGBoost: **V0** chỉ feature giao dịch (= baseline "XGBoost thuần", sàn ~19.75 F1) → **V1** + điểm mule (2 số vô hướng) → **V2** + đủ embedding 128 chiều (pipeline đề xuất). Delta V0→V2 = con số chứng minh đóng góp 1; delta V1→V2 = embedding dày có hơn mỗi điểm mule vô hướng không (nếu không hơn cũng là phát hiện đáng nói).
     - **Grouped SHAP (1 hình minh họa):** TreeSHAP trên XGBoost V2, gộp |SHAP| theo nhóm [feature giao dịch] / [emb gửi 64] / [emb nhận 64] / [điểm mule]. **Hai bẫy:** (1) lệch theo số cột — emb 128 cột gần như chắc lớn hơn giao dịch ~25 cột chỉ vì nhiều cột, đừng kết luận từ tổng thô; (2) tương quan emb↔feature thô làm TreeSHAP tán tín dụng → SHAP chỉ gợi ý, không phải phân rã nhân quả. Tính trên mẫu (đủ ca dương), không tính trên 1.3M dòng. → đây là lý do ablation là bằng chứng chính (miễn nhiễm số cột), SHAP là phụ.

2. **Hệ thống phát hiện gian lận real-time hai đường — chịu tải (ĐÓNG GÓP 2 — LÀM THẬT, đo đạc):** trục đo KHÁC đóng góp 1 — latency/throughput dưới tải, không phải PR-AUC. Real-time hợp lệ = xử lý theo **sự kiện** (không gom lô), đạt **SLA latency công bố trước** (vd p99 < ngưỡng X ms, X neo vào nghiệp vụ cổng thanh toán), ổn định dưới tải.
   - **Đường nóng (mỗi giao dịch, PHẢI đạt SLA — chạy CPU, KHÔNG đụng GPU):** giao dịch tới → tra embedding từ cache RAM (Redis/dict in-process) → ghép feature → XGBoost predict → điểm fraud. Đây là đường quyết định "có real-time hay không". Tối ưu: cache lookup O(1), tránh allocate trong vòng nóng, serialize gọn.
   - **Đường nguội (chạy nền, bất đồng bộ, KHÔNG nằm trong SLA — dùng GPU):** GraphSAGE **inductive inference** cập nhật embedding cho tài khoản bị ảnh hưởng (sample k-hop `[15,10]`), ghi đè cache. Không chặn đường nóng. Không train lại. Embedding có thể hơi cũ → sinh ra đường cong staleness.
   - **Đo chịu tải:** bơm ở nhiều mức throughput (vd 500 / 2000 / 5000 tx/s) → vẽ đường latency-vs-throughput, tìm điểm bão hòa và nút thắt; kiểm backpressure khi consumer chậm hơn producer; chạy dài kiểm memory leak. Harness đo p50/p95/p99 + profiler (đo trước, tối ưu sau).
   - **So trực tiếp hợp lệ:** chạy lại GFP (Snap ML) trên CHÍNH máy này → so cùng phần cứng, khác phương pháp. Số công bố GFP/RIPPLE++/NVIDIA (máy khác) chỉ để định vị, ghi rõ cách biệt do phần cứng.
   - **Thí nghiệm ăn điểm systems:** đường cong staleness — refresh embedding mỗi 1h/6h/1 ngày vs tức thì ảnh hưởng F1/recall bao nhiêu; tìm điểm "đủ tốt". Novelty (chưa thấy ai đo trên IBM AML — tự verify lit review).
   - **Nút thắt (RTX 2050 4GB / 16GB RAM):** đồ thị ở RAM không phải VRAM; đồ thị động tự quản adjacency (dict/CSR) — engineering nặng nhất; giới hạn fanout chống hub nổ; cold-start node mới (fallback dựa feature).
   - **Framing trung thực:** đạt SLA dưới tải vừa phải là real-time hợp lệ; KHÔNG hứa throughput ngang server (GFP 12–23k tx/s). Phát biểu: "real-time đạt SLA p99 < X ms tới throughput Y tx/s trên phần cứng giới hạn; điểm bão hòa và nút thắt là Z". Điều kiện không-gian-lận: đường nóng KHÔNG phụ thuộc GPU (XGBoost chạy CPU — đã chốt).
   - **Tiêu chí thành công = ĐO + PHÂN TÍCH được** (không nhị phân đạt/thất bại theo mốc tốc độ): chạy end-to-end, đạt SLA dưới tải, có đường latency-throughput, có đường cong staleness, phân tích được nút thắt.
   - **Quản trị rủi ro:** làm xong + chắc đóng góp 1 TRƯỚC, rồi mới sang đóng góp 2 (mini-project systems, dễ ngốn gấp đôi thời gian).

> **Đã bỏ khỏi đóng góp:** "đo độ nhạy / protocol-sensitivity" (khung cũ 2026-06-14). Lý do ở mục 7. Phần giá trị còn lại (so temporal vs random) giữ làm thí nghiệm phụ trong đánh giá đóng góp 1, không phải đóng góp riêng.

### Baseline bắt buộc cho đóng góp 1

| Baseline                               | Vai trò                                           | Chi phí |
| -------------------------------------- | ------------------------------------------------- | ------- |
| XGBoost thuần (feature giao dịch)      | sàn dưới (GFP paper: ~19.75 F1)                   | CPU     |
| GFP (Snap ML) + XGBoost                | đối thủ chính cùng triết lý "graph feature + GBM" | CPU     |
| GraphSAGE embedding + XGBoost (đề tài) | pipeline đề xuất                                  | GPU 4GB |
| GIN/PNA/Multi-PNA edge-level           | chỉ trích dẫn số công bố, không tự chạy           | —       |

### Việc rút ra cho bước hiện tại (chia dữ liệu)

- **Split chính = cắt-index chuẩn Altman.** Mọi kết quả chính của đóng góp 1 báo cáo trên split này.
- Split-day giữ lại làm baseline cho đóng góp 2 (real-time refresh theo ngày), không dùng báo cáo metric chính.
- Mọi kết quả xuất ra phải ghi rõ split.

---
