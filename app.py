import time
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import paths
from metrics import find_best_threshold

MODEL_PATH = "models/baseline_lgb_seed0.joblib"
RAW_CSV = f"{paths.RAW_DIR}/HI-Small_Trans_split_index.csv"
PATTERNS_TXT = f"{paths.RAW_DIR}/HI-Small_Patterns.txt"
DISPLAY_COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
PATTERN_KEY = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
               "Amount Paid", "Payment Format"]
WINDOW = 50

PATTERN_EXPLAIN = {
    "FAN-OUT": "Một tài khoản nguồn chuyển tiền tỏa ra rất nhiều tài khoản nhận khác nhau trong "
               "thời gian ngắn. Đây là kỹ thuật 'chia nhỏ' (structuring) để mỗi giao dịch có giá trị "
               "nhỏ, tránh vượt ngưỡng báo cáo và tránh để một tài khoản nhận giữ toàn bộ số tiền lớn.",
    "FAN-IN": "Rất nhiều tài khoản nguồn khác nhau cùng chuyển tiền dồn về một tài khoản nhận duy nhất. "
              "Đây là bước 'gom tiền' (placement/collection) — tài khoản nhận thường đóng vai trò tài "
              "khoản trung chuyển (mule) để tập trung tiền trước khi chuyển tiếp hoặc rút ra.",
    "CYCLE": "Tiền được chuyển qua một chuỗi tài khoản trung gian rồi quay trở lại tài khoản gần với "
             "nguồn ban đầu. Vòng lặp này tạo lớp giao dịch hợp lệ giả (layering) khiến nguồn gốc thật "
             "của tiền khó truy vết dù dòng tiền cuối cùng không đổi chủ thực sự.",
    "GATHER-SCATTER": "Tiền được gom lại từ nhiều tài khoản nguồn vào một (hoặc vài) tài khoản trung "
                       "gian (gather), sau đó lập tức tỏa ra nhiều tài khoản khác (scatter). Kết hợp cả "
                       "gom lẫn chia nhỏ khiến việc lần theo dòng tiền gốc gần như bất khả thi.",
    "RANDOM": "Giao dịch nằm trong một chuỗi chuyển tiền gian lận nhưng không theo cấu trúc đồ thị cố "
              "định (không rõ fan-out/fan-in/cycle/gather-scatter) — mô phỏng hành vi rửa tiền 'nhiễu', "
              "cố tình tránh tạo dấu hiệu cấu trúc để né các luật phát hiện dựa trên topology.",
    "STACK": "Tiền được chuyển tuần tự qua một chuỗi tài khoản trung gian xếp lớp lên nhau (không quay "
             "lại tài khoản nguồn như cycle, không tỏa/gom như fan-out/fan-in). Đây là kỹ thuật 'xếp "
             "lớp' (layering) kinh điển: mỗi lớp trung gian làm mờ dần liên kết với nguồn gốc ban đầu.",
    "BIPARTITE": "Một nhóm nhiều tài khoản nguồn cùng chuyển tiền chằng chịt tới một nhóm nhiều tài "
                 "khoản nhận khác (quan hệ nhiều-đến-nhiều), tạo mạng lưới giao dịch dày đặc khiến khó "
                 "phân biệt luồng tiền thật với các giao dịch dùng để ngụy trang.",
    "SCATTER-GATHER": "Tiền từ một tài khoản tỏa ra nhiều tài khoản trung gian (scatter), sau đó các "
                       "tài khoản này cùng gom tiền lại vào một tài khoản đích (gather) — ngược thứ tự "
                       "với gather-scatter. Chia nhỏ số tiền lớn để né ngưỡng báo cáo rồi tái hợp nó ở "
                       "nơi khác.",
}

st.set_page_config(page_title="Fraud detection", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_feature_matrix(split):
    df = pd.read_parquet(paths.TXN_MATRIX.format(split))
    y = df.pop("Is Laundering").to_numpy()
    return df, y


@st.cache_data
def compute_threshold(_model):
    Xv, yv = load_feature_matrix("val")
    sv = _model.predict_proba(Xv)[:, 1]
    return find_best_threshold(yv, sv)


@st.cache_data
def load_patterns():
    cols = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
            "Payment Format", "Is Laundering"]
    rows, pattern, description, attempt_id = [], None, None, -1
    with open(PATTERNS_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt_id += 1
                header = line.split(" - ", 1)[1]
                if ":" in header:
                    pattern, description = header.split(":", 1)
                    pattern, description = pattern.strip(), description.strip()
                else:
                    pattern, description = header.strip(), ""
            elif line.startswith("END LAUNDERING ATTEMPT"):
                pattern, description = None, None
            elif line.strip():
                parts = line.split(",")
                if len(parts) == len(cols):
                    d = dict(zip(cols, parts))
                    d["pattern"] = pattern
                    d["description"] = description
                    d["attempt_id"] = attempt_id
                    rows.append(d)
    pat = pd.DataFrame(rows)
    pat["Amount Paid"] = pat["Amount Paid"].astype(float)
    return pat[PATTERN_KEY + ["pattern", "description", "attempt_id"]]


@st.cache_data
def load_test_with_scores(_model):
    X, y = load_feature_matrix("test")
    score = _model.predict_proba(X)[:, 1]

    dt = {"From Bank": str, "Account": str, "To Bank": str, "Account.1": str, "split": str}
    raw = pd.read_csv(RAW_CSV, dtype=dt, usecols=DISPLAY_COLS + ["split"])
    raw = raw[raw["split"] == "test"].reset_index(drop=True)
    assert len(raw) == len(X), f"lệch dòng: csv test {len(raw)} vs txn_matrix test {len(X)}"
    assert (raw["Is Laundering"].to_numpy() == y).all(), "lệch nhãn giữa csv và txn_matrix -> thứ tự dòng không khớp"

    raw = raw.merge(load_patterns(), on=PATTERN_KEY, how="left")
    raw["pattern"] = raw["pattern"].fillna("—")
    raw["description"] = raw["description"].fillna("")
    raw["attempt_id"] = raw["attempt_id"].fillna(-1).astype(int)

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    raw["score"] = score
    raw["row_id"] = np.arange(len(raw))          # để tra ngược X sau khi sort theo thời gian
    raw["src_dest"] = (raw["From Bank"] + "|" + raw["Account"]
                        + " -> " + raw["To Bank"] + "|" + raw["Account.1"])
    raw = raw.sort_values("Timestamp").reset_index(drop=True)
    return raw, X


@st.cache_data
def group_means(_model):
    """Trung bình từng đặc trưng của nhóm gian lận vs nhóm bình thường trên tập test."""
    X, y = load_feature_matrix("test")
    return X.assign(is_laundering=y).groupby("is_laundering").mean()


model = load_model()
explainer = load_explainer(model)
threshold = compute_threshold(model)
raw, X = load_test_with_scores(model)
raw["cảnh báo"] = np.where(raw["score"] >= threshold, "GIAN LẬN", "bình thường")
gmeans = group_means(model)

st.title("Demo phát hiện giao dịch rửa tiền")
st.caption(f"Model: baseline_lgb_seed0 (nhóm 1, LightGBM, 90 đặc trưng) | "
           f"Ngưỡng tune trên val: {threshold:.4f} | Tập test: {len(raw):,} giao dịch, "
           f"{int(raw['Is Laundering'].sum()):,} gian lận")

if "pos" not in st.session_state:
    st.session_state.pos = 0

alert_filter = st.selectbox("Lọc theo cảnh báo", ["Tất cả", "GIAN LẬN", "bình thường"])

raw_view = raw
if alert_filter != "Tất cả":
    raw_view = raw_view[raw_view["cảnh báo"] == alert_filter]
raw_view = raw_view.reset_index(drop=True)

max_pos = max(len(raw_view) - WINDOW, 0)
if st.session_state.pos > max_pos:
    st.session_state.pos = 0

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    pos = st.slider("Vị trí trong dòng thời gian (theo Timestamp)", 0,
                     max_pos, st.session_state.pos, step=WINDOW)
    st.session_state.pos = pos
with c2:
    autoplay = st.checkbox("▶ Tự động phát (giả lập real-time)")
with c3:
    speed = st.select_slider("Tốc độ (giây/lần)", options=[0.2, 0.5, 1.0, 2.0], value=0.5)

window = raw_view.iloc[pos:pos + WINDOW].copy()

st.dataframe(
    window[["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Paid", "Payment Currency", "score", "cảnh báo", "Is Laundering", "src_dest"]],
    use_container_width=True, height=350,
)

st.subheader("Xem chi tiết một giao dịch")
options = window["row_id"].tolist()
labels = {r: f"{sd} | {t} | {a:,.0f} {c}"
          for r, t, a, sd, c in zip(window["row_id"], window["Timestamp"],
                                     window["Amount Paid"], window["src_dest"], window["Payment Currency"])}
picked = st.selectbox("Chọn giao dịch", options, format_func=lambda r: labels[r])

if st.button("Xem chi tiết") and picked is not None:
    row_x = X.iloc[[picked]]
    row_raw = raw[raw["row_id"] == picked].iloc[0]
    score = float(model.predict_proba(row_x)[0, 1])
    pred = "GIAN LẬN" if score >= threshold else "bình thường"

    st.markdown(f"**Điểm dự đoán: {score:.4f}** (ngưỡng {threshold:.4f}) → dự đoán **{pred}**, "
                f"nhãn thật: {'GIAN LẬN' if row_raw['Is Laundering'] == 1 else 'bình thường'}")

    if row_raw["Is Laundering"] == 1:
        st.markdown(f"### Vì sao đây là giao dịch gian lận — kiểu pattern: `{row_raw['pattern']}`")
        if row_raw["description"]:
            st.caption(f"Mô tả gốc trong dataset (HI-Small_Patterns.txt): {row_raw['description']}")
        st.write(PATTERN_EXPLAIN.get(
            row_raw["pattern"],
            "Giao dịch được gắn nhãn gian lận nhưng không xác định được kiểu pattern cụ thể trong dataset."))

        related = raw[(raw["attempt_id"] == row_raw["attempt_id"]) & (raw["row_id"] != picked)]
        if len(related):
            st.write(f"**{len(related)} giao dịch khác thuộc cùng chuỗi rửa tiền này "
                     f"({row_raw['pattern']}):**")
            st.dataframe(
                related[["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                         "Amount Paid", "Payment Currency", "score", "cảnh báo", "src_dest"]],
                use_container_width=True,
            )
        else:
            st.caption("Không tìm thấy giao dịch liên quan nào khác trong tập test (attempt chỉ có "
                       "1 giao dịch trong test, hoặc phần còn lại nằm ở tập train/val).")
    else:
        st.caption("Giao dịch này không thuộc pattern rửa tiền nào đã biết trong dataset.")

    sv = explainer.shap_values(row_x)[0]
    order = np.argsort(-np.abs(sv))[:10]
    top_cols = row_x.columns[order]

    imp_df = pd.DataFrame({
        "đặc trưng": top_cols,
        "đóng góp SHAP (đẩy về gian lận nếu dương)": sv[order],
        "giá trị giao dịch này": row_x.iloc[0][top_cols].to_numpy(),
        "TB nhóm gian lận (test)": [gmeans.loc[1, c] if c in gmeans.columns else np.nan for c in top_cols],
        "TB nhóm bình thường (test)": [gmeans.loc[0, c] if c in gmeans.columns else np.nan for c in top_cols],
    })
    st.write("**10 đặc trưng ảnh hưởng nhiều nhất đến điểm số của đúng giao dịch này** "
             "(SHAP dương = đẩy điểm về phía gian lận):")
    st.dataframe(imp_df.style.format({
        "đóng góp SHAP (đẩy về gian lận nếu dương)": "{:+.4f}",
        "giá trị giao dịch này": "{:.3f}",
        "TB nhóm gian lận (test)": "{:.3f}",
        "TB nhóm bình thường (test)": "{:.3f}",
    }), use_container_width=True)

    st.write("**Biểu đồ biểu thị độ quan trọng của đặc trưng")
    gain = model.booster_.feature_importance(importance_type="gain")
    gdf = pd.DataFrame({"đặc trưng": X.columns, "gain": gain}).sort_values("gain", ascending=False).head(15)
    st.bar_chart(gdf.set_index("đặc trưng"))

if autoplay:
    time.sleep(speed)
    st.session_state.pos = min(st.session_state.pos + WINDOW, max_pos)
    st.rerun()